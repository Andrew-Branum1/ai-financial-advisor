import sys
import os
import argparse

# Add project root to the path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.test_overfitting import run_evaluation
from config import MODEL_CONFIGS

def find_latest_model(config_name: str) -> str | None:
    """Finds the path to the most recently trained model for a given config."""
    models_dir = "models"
    if not os.path.isdir(models_dir):
        print(f"Warning: Models directory '{models_dir}' not found.")
        return None

    # Find all directories for this model config, sorted by most recent
    subfolders = [f for f in os.listdir(models_dir) if f.startswith(config_name) and os.path.isdir(os.path.join(models_dir, f))]
    if not subfolders:
        return None
        
    latest_folder = sorted(subfolders, reverse=True)[0]
    model_path = os.path.join(models_dir, latest_folder, "model.zip")
    
    if os.path.exists(model_path):
        return model_path
    
    print(f"Warning: model.zip not found in the latest directory: {os.path.join(models_dir, latest_folder)}")
    return None

def validate_models(models_to_validate: str):
    """Iterates through model configs, finds the latest model, and evaluates it."""
    print(f"\n--- Starting Validation for: {models_to_validate.upper()} Models ---")
    
    if models_to_validate == 'all':
        target_configs = list(MODEL_CONFIGS.keys())
    else:
        target_configs = [m for m in MODEL_CONFIGS.keys() if models_to_validate in m]

    if not target_configs:
        print(f"No models found matching the filter: '{models_to_validate}'")
        return

    models_found = 0
    for config_name in target_configs:
        print(f"\nSearching for latest model for: '{config_name}'")
        latest_model_path = find_latest_model(config_name)
        
        if latest_model_path:
            models_found += 1
            run_evaluation(latest_model_path, config_name)
        else:
            print(f"-> No trained model found for '{config_name}'. Skipping.")
            
    print(f"\n--- Validation Complete ---")
    print(f"Evaluated {models_found} out of {len(target_configs)} selected models.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate AI Financial Advisor models.")
    parser.add_argument(
        "--models", 
        type=str, 
        choices=["short_term", "long_term", "all"], 
        default="all", 
        help="Which set of models to validate."
    )
    args = parser.parse_args()
    
    validate_models(models_to_validate=args.models)