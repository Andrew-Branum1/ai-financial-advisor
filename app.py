# app.py
#
# This is the main Flask application that serves the AI Financial Advisor.
# It handles the web interface, loads our trained RL models, and uses the LLM
# to generate financial advice based on the model's predictions.

import os
import json
import logging
import sqlite3
from glob import glob

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from stable_baselines3 import PPO

# --- Project-specific Imports ---
import config
# This will now work correctly because the new advisor.py provides the instance.
from llm.advisor import financial_advisor
# We will patch the data loader function from src/utils below.
from src.utils import load_market_data_from_db

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# --- Database Hotfix ---
# This patch ensures app.py uses the same corrected database logic as the training script.
def patched_load_market_data_from_db():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'market_data.db')
    logging.info(f"Attempting to load all data from wide-format table in DB: {db_path}")
    try:
        with sqlite3.connect(db_path) as conn:
            query = 'SELECT * FROM features_market_data ORDER BY "Date" ASC'
            df = pd.read_sql_query(query, conn)
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        return None
    
    if 'Date' not in df.columns:
        logging.error(f"CRITICAL: 'Date' column not found. Columns are: {df.columns.tolist()}")
        return None
        
    df.rename(columns={'Date': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# Replace the original function from src/utils with our patched version
load_market_data_from_db = patched_load_market_data_from_db
logging.info("Applied runtime patch to 'load_market_data_from_db'.")
# --- End Hotfix ---


# --- Global Cache for Models ---
MODEL_CACHE = {}

def find_latest_model(model_name_prefix: str) -> str | None:
    search_path = os.path.join('models', f'{model_name_prefix}_*')
    list_of_dirs = glob(search_path)
    if not list_of_dirs: return None
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    model_path = os.path.join(latest_dir, 'model.zip')
    return model_path if os.path.exists(model_path) else None

def load_all_models():
    logging.info("Starting to load all available models...")
    for model_name in config.MODEL_CONFIGS.keys():
        logging.info(f"Searching for model: {model_name}")
        model_path = find_latest_model(model_name)
        if model_path:
            try:
                MODEL_CACHE[model_name] = PPO.load(model_path, device='cpu')
                logging.info(f"✅ Successfully loaded model '{model_name}' from {model_path}")
            except Exception as e:
                logging.error(f"❌ Failed to load model '{model_name}': {e}")
        else:
            logging.warning(f"⚠️ No trained model found for '{model_name}'.")
    logging.info("Model loading process complete.")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_advice', methods=['POST'])
def get_advice():
    data = request.get_json()
    if not data: return jsonify({'error': 'Invalid request.'}), 400

    model_name = data.get('model_name')
    investment_amount = float(data.get('investment_amount', 10000))
    user_age = int(data.get('age', 30))

    if not model_name: return jsonify({'error': 'No model name provided.'}), 400

    logging.info(f"Request for {model_name} with amount ${investment_amount:,.2f} for age {user_age}")

    model = MODEL_CACHE.get(model_name)
    if not model: return jsonify({'error': f'Model "{model_name}" is not available.'}), 404

    try:
        model_folder = os.path.dirname(find_latest_model(model_name))
        info_path = os.path.join(model_folder, 'training_info.json')
        with open(info_path, 'r') as f:
            training_info = json.load(f)
        window_size = training_info['best_hyperparameters'].get('window_size', 60)
        
        model_cfg = config.MODEL_CONFIGS[model_name]
        
        df = load_market_data_from_db()
        if df is None or df.empty:
             raise ValueError("Failed to load market data from DB.")
        
        # --- CORE FIX for IndexError ---
        # Dynamically determine the exact list of tickers this model was trained on.
        features_for_observation = model_cfg['features_to_use']
        
        observation_columns_for_model = [f"{ticker}_{feat}" for ticker in config.ALL_TICKERS for feat in features_for_observation]
        available_model_cols = [col for col in observation_columns_for_model if col in df.columns]
        
        # This is the crucial step: derive the tickers from the available data columns.
        tickers_in_model = sorted(list(set([col.split('_')[0] for col in available_model_cols])))
        
        # Re-build the observation columns based on the tickers that are *actually* available.
        final_observation_cols = [f"{ticker}_{feat}" for ticker in tickers_in_model for feat in features_for_observation]
        observation = df[final_observation_cols].tail(window_size).values
        # --- END FIX ---

    except Exception as e:
        logging.error(f"Data preparation error for model {model_name}: {e}", exc_info=True)
        return jsonify({'error': 'Could not prepare data for the model.'}), 500

    action, _ = model.predict(observation, deterministic=True)
    predicted_weights = np.maximum(action, 0)
    if np.sum(predicted_weights) > 0:
        predicted_weights /= np.sum(predicted_weights)

    portfolio = []
    latest_prices = df[[f"{ticker}_close" for ticker in tickers_in_model if f"{ticker}_close" in df.columns]].iloc[-1]
    
    # --- FIX THE LOOP: Iterate over the dynamically determined tickers_in_model ---
    for i, ticker in enumerate(tickers_in_model):
        # Now the length of this loop matches the length of predicted_weights
        weight = predicted_weights[i]
        price_col = f"{ticker}_close"
        if weight > 0.001 and price_col in latest_prices:
            allocated_dollars = investment_amount * weight
            price = latest_prices[price_col]
            num_shares = int(allocated_dollars // price)

            if num_shares > 0:
                portfolio.append({
                    'ticker': ticker,
                    'shares': num_shares,
                    'price': f"${price:,.2f}",
                    'value': f"${num_shares * price:,.2f}",
                    'weight': f"{weight:.1%}"
                })

    try:
        rationale = financial_advisor.generate_advice(
            user_profile={
                'risk_tolerance': model_name.split('_')[-1],
                'time_horizon': model_name.split('_')[0],
                'age': user_age,
                'investment_amount': f"${investment_amount:,.2f}"
            },
            portfolio_allocation={item['ticker']: item['weight'] for item in portfolio}
        )
    except Exception as e:
        logging.error(f"LLM advice generation failed: {e}")
        rationale = "Could not generate a detailed rationale at this time."

    return jsonify({'portfolio': portfolio, 'rationale': rationale})


if __name__ == '__main__':
    load_all_models()
    app.run(debug=False, port=5000)
