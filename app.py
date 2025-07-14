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
import traceback

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from stable_baselines3 import PPO

# --- Project-specific Imports ---
import config
from llm.advisor import financial_advisor
# We will patch the data loader function from src/utils below.
from src.data_manager import load_market_data_from_db

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)


# --- Global Cache for Models ---
MODEL_CACHE = {}

def find_latest_model(model_name_prefix: str) -> str | None:
    """Finds the most recent model zip file for a given config."""
    search_path = os.path.join('models', f'{model_name_prefix}_*')
    list_of_dirs = glob(search_path)
    if not list_of_dirs: return None
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    model_path = os.path.join(latest_dir, 'model.zip')
    return model_path if os.path.exists(model_path) else None

def load_all_models():
    """Loads the latest version of each model specified in config.py into memory."""
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
    """Renders the main web page."""
    # Pass the list of available models to the frontend template
    available_models = sorted(list(MODEL_CACHE.keys()))
    return render_template('index.html', models=available_models)


@app.route('/get_advice', methods=['POST'])
def get_advice():
    """Handles the API request for financial advice."""
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
        # --- Prepare data exactly as the model expects ---
        model_folder = os.path.dirname(find_latest_model(model_name))
        info_path = os.path.join(model_folder, 'training_info.json')
        with open(info_path, 'r') as f:
            training_info = json.load(f)

        window_size = training_info['best_hyperparameters'].get('window_size', 60)
        features_from_training = training_info['features_used']

        df = load_market_data_from_db()
        df.index = pd.to_datetime(df.index)
        if df is None or df.empty:
             raise ValueError("Failed to load market data from DB.")

        # --- BUG FIX STARTS HERE ---
        # The definitive list of tickers must be derived from the actual columns
        # available in the data that are also part of the model's feature set.
        initial_observation_cols = [f"{t}_{f}" for t in config.ALL_TICKERS for f in features_from_training]
        final_observation_cols = [c for c in initial_observation_cols if c in df.columns]

        # This is now the source of truth for the tickers the model will predict on.
        # It perfectly matches the structure of the 'observation' tensor.
        final_tickers_for_model = sorted(list(set([c.split('_')[0] for c in final_observation_cols])))
        # --- BUG FIX ENDS HERE ---

        # Get the most recent data for the observation
        observation = df[final_observation_cols].tail(window_size).values

    except Exception as e:
        logging.error(f"Data preparation error for model {model_name}: {e}", exc_info=True)
        return jsonify({'error': 'Could not prepare data for the model.'}), 500

    # --- Get Model Prediction ---
    action, _ = model.predict(observation, deterministic=True)

    # Unpack and clean the action
    if isinstance(action, tuple) and len(action) > 0:
        predicted_weights = np.array(action[0]).flatten()
    else:
        predicted_weights = np.array(action).flatten()

    # Normalize weights to ensure they sum to 1 (or less if holding cash)
    if np.sum(predicted_weights) > 1:
        predicted_weights /= np.sum(predicted_weights)

    cash_weight = 1.0 - np.sum(predicted_weights)

    # --- Format Portfolio for Display ---
    portfolio = []
    existing_price_cols = [f"{t}_close" for t in final_tickers_for_model if f"{t}_close" in df.columns]
    latest_prices = df[existing_price_cols].iloc[-1]

    # Iterate over the definitive list of tickers that matches the prediction weights
    for i, ticker in enumerate(final_tickers_for_model):
        # This is now safe from the IndexError
        weight = predicted_weights[i]
        price_col = f"{ticker}_close"

        if weight > 0.001:
            # This safety check prevents the KeyError
            if price_col in latest_prices.index:
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
            else:
                logging.warning(f"Could not find price for ticker '{ticker}'. Skipping it in portfolio display.")

    if cash_weight > 0.001:
        portfolio.append({
            'ticker': 'Cash', 'shares': '-', 'price': '$1.00',
            'value': f"${investment_amount * cash_weight:,.2f}",
            'weight': f"{cash_weight:.1%}"
        })

    # --- Generate LLM Rationale ---
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
    app.run(debug=True, port=5000)