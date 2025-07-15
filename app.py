import os
import json
import sqlite3
from glob import glob

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from stable_baselines3 import PPO

import config
from llm.advisor import financial_advisor
from src.data_manager import load_market_data

app = Flask(__name__)

# A simple cache to hold loaded models in memory.
MODEL_CACHE = {}

def _find_latest_model_path(model_name_prefix: str) -> str | None:
    """Finds the path to the most recent model file for a given name."""
    search_path = os.path.join('models', f'{model_name_prefix}_*')
    list_of_dirs = glob(search_path)
    if not list_of_dirs:
        return None
    
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    model_path = os.path.join(latest_dir, 'model.zip')
    
    return model_path if os.path.exists(model_path) else None

def _load_all_models():
    """Loads all models specified in the config into the cache."""
    print("Loading all available models...")
    for model_name in config.MODEL_CONFIGS.keys():
        model_path = _find_latest_model_path(model_name)
        if model_path:
            try:
                MODEL_CACHE[model_name] = PPO.load(model_path, device='cpu')
                print(f"-> Successfully loaded model '{model_name}'")
            except Exception as e:
                print(f"-> ERROR: Failed to load model '{model_name}': {e}")
        else:
            print(f"-> WARNING: No trained model found for '{model_name}'.")
    print("Model loading complete.")


@app.route('/')
def home():
    """Renders the main web page with a list of available models."""
    available_models = sorted(list(MODEL_CACHE.keys()))
    return render_template('index.html', models=available_models)


@app.route('/get_advice', methods=['POST'])
def get_advice():
    """Handles the API request for generating financial advice."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request.'}), 400

    model_name = data.get('model_name')
    investment_amount = float(data.get('investment_amount', 10000))
    user_age = int(data.get('age', 30))

    if not model_name:
        return jsonify({'error': 'No model name provided.'}), 400

    model = MODEL_CACHE.get(model_name)
    if not model:
        return jsonify({'error': f'Model "{model_name}" is not available.'}), 404

    try:
        # Prepare data exactly as the model expects
        model_folder = os.path.dirname(_find_latest_model_path(model_name))
        with open(os.path.join(model_folder, 'training_info.json'), 'r') as f:
            training_info = json.load(f)

        window_size = training_info['best_hyperparameters'].get('window_size', 60)
        features_from_training = training_info['features_used']

        df = load_market_data()
        if df is None or df.empty:
             raise ValueError("Failed to load market data from DB.")
        df.index = pd.to_datetime(df.index)

        # Determine the exact feature columns the model was trained on
        initial_observation_cols = [f"{t}_{f}" for t in config.ALL_TICKERS for f in features_from_training]
        final_observation_cols = [c for c in initial_observation_cols if c in df.columns]
        
        # This determines the tickers the model will predict on, matching the observation tensor
        final_tickers_for_model = sorted(list(set([c.split('_')[0] for c in final_observation_cols])))
        
        # Get the most recent data for the observation
        observation = df[final_observation_cols].tail(window_size).values

    except Exception as e:
        print(f"Data preparation error for model {model_name}: {e}")
        return jsonify({'error': 'Could not prepare data for the model.'}), 500

    # Get model's prediction
    action, _ = model.predict(observation, deterministic=True)
    predicted_weights = np.array(action).flatten()

    # Normalize weights to ensure they sum to 1
    if np.sum(predicted_weights) > 1:
        predicted_weights /= np.sum(predicted_weights)
    cash_weight = 1.0 - np.sum(predicted_weights)

    # Format the portfolio for display
    portfolio = []
    latest_prices = df[[f"{t}_close" for t in final_tickers_for_model if f"{t}_close" in df.columns]].iloc[-1]

    for i, ticker in enumerate(final_tickers_for_model):
        weight = predicted_weights[i]
        price_col = f"{ticker}_close"
        if weight > 0.001 and price_col in latest_prices.index:
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

    if cash_weight > 0.001:
        portfolio.append({
            'ticker': 'Cash', 'shares': '-', 'price': '$1.00',
            'value': f"${investment_amount * cash_weight:,.2f}",
            'weight': f"{cash_weight:.1%}"
        })

    # Generate the text-based advice from the LLM
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
        print(f"LLM advice generation failed: {e}")
        rationale = "Could not generate a detailed rationale at this time."

    return jsonify({'portfolio': portfolio, 'rationale': rationale})


if __name__ == '__main__':
    _load_all_models()
    app.run(debug=True, port=5000)