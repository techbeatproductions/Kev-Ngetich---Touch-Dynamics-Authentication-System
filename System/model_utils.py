import logging
import os
import glob
import joblib

# Get the base directory where models are saved
models_base_directory = r'C:\Users\ngeti\Documents\4.2\Final Year Project System\models'

def load_model(model_name):
    # Get the latest timestamped directory
    latest_dir = max(glob.glob(os.path.join(models_base_directory, '*')), key=os.path.getctime)

    # Construct the model path
    model_path = os.path.join(latest_dir, f"best_{model_name}_model.pkl")

    if os.path.exists(model_path):
        logging.info(f"Loading model from file: {model_path}")
        try:
            model = joblib.load(model_path)
            logging.info(f"Model loaded successfully: {model_name}")
            return model
        except Exception as e:
            logging.error(f"Error loading model '{model_name}': {e}")
            return None
    else:
        logging.error(f"Model '{model_name}' not found")
        return None