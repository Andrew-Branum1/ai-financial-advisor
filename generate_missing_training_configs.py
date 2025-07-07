import os
import json

# Import features from config files
try:
    from config_long_term import FEATURES_TO_USE_IN_MODEL as LT_FEATURES
except ImportError:
    LT_FEATURES = []
try:
    from config_short_term import FEATURES_TO_USE_IN_MODEL as ST_FEATURES
except ImportError:
    ST_FEATURES = []

def main():
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found.")
        return
    for subdir in os.listdir(models_dir):
        model_path = os.path.join(models_dir, subdir)
        if not os.path.isdir(model_path):
            continue
        config_path = os.path.join(model_path, 'training_config.json')
        if os.path.exists(config_path):
            print(f"Config already exists: {config_path}")
            continue
        # Determine type and defaults
        if subdir.startswith('long_term'):
            window_size = 60
            features = LT_FEATURES
        elif subdir.startswith('short_term'):
            window_size = 30
            features = ST_FEATURES
        else:
            print(f"Unknown model type for {subdir}, skipping.")
            continue
        config = {
            "env_params": {"window_size": window_size},
            "ppo_params": {"features_to_use": features}
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created config: {config_path}")

if __name__ == "__main__":
    main() 