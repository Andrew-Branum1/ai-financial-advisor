import os
import sys
from eval.test_overfitting import run_evaluation
from config import MODEL_CONFIGS
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_latest_model(config_name: str) -> str | None:
    models_dir = "models"
    if not os.path.isdir(models_dir):
        print(f"'{models_dir}' not found.")
        return None

    subfolders = [f for f in os.listdir(models_dir) if f.startswith(config_name) and os.path.isdir(os.path.join(models_dir, f))]
    if not subfolders:
        return None
        
    latest_folder = sorted(subfolders, reverse=True)[0]
    model_path = os.path.join(models_dir, latest_folder, "model.zip")
    
    if os.path.exists(model_path):
        return model_path
    

def validate_all_models():
    print("\nSTARTING")
    target_configs = list(MODEL_CONFIGS.keys())
    for config_name in target_configs:
        latest_model_path = find_latest_model(config_name)
        if latest_model_path:
            run_evaluation(latest_model_path, config_name)
        else:
            print("not found")
            

validate_all_models()