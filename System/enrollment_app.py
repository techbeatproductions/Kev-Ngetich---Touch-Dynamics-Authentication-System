from flask import Flask, jsonify, request
import logging
import os
import joblib
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from train_and_save_models import train_svdd, enroll_user  # Import your function directly

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename='enroll_user_log.txt',  # Name of the log file
    level=logging.DEBUG,             # Log level: DEBUG captures all levels of logs
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

# Directories for saving models and user profiles
user_profiles_directory = r'C:\Users\ngeti\Documents\4.2\Final Year Project System\user_profiles'
models_base_directory = r'C:\Users\ngeti\Documents\4.2\Final Year Project System\models'
plots_base_directory = r'C:\Users\ngeti\Documents\4.2\Final Year Project System\plots'


# Ensure directories exist
os.makedirs(user_profiles_directory, exist_ok=True)
os.makedirs(models_base_directory, exist_ok=True)
os.makedirs(plots_base_directory, exist_ok=True)

def save_user_profile(user_id, user_profile):
    """
    Save the user profile to a pickle file.
    """
    # Get the most recent time-stamped folder
    most_recent_folder = max(glob.glob(os.path.join(models_base_directory, '*')), key=os.path.getctime)
    user_profile_path = os.path.join(most_recent_folder, f"{user_id}_profile.pkl")
    print(f"Saving user profile to: {user_profile_path}")
    joblib.dump(user_profile, user_profile_path)
    logging.info(f"User profile saved to: {user_profile_path}")

@app.route('/create_profile', methods=['POST'])
def create_profile():
    try:
        data = request.json
        user_id = data['user_id']
        touch_data = data['touch_data']
        
        # Convert touch_data to DataFrame
        df = pd.DataFrame(touch_data)
        
        # Validate the DataFrame has at least 6 columns
        if df.shape[1] < 6:
            logging.error(f"Touch data for user {user_id} has insufficient columns: {df.shape[1]}")
            return jsonify({"message": "Touch data must have at least 6 columns"}), 400
        
        logging.info(f"Touch data shape for user {user_id}: {df.shape}")
        
        # Save the raw touch data as CSV (without headers)
        user_data_path = os.path.join(user_profiles_directory, f"{user_id}_raw.csv")
        df.to_csv(user_data_path, index=False, header=False)
        logging.info(f"Raw touch data saved to: {user_data_path}")
        
        # Define best SVDD parameters from previous training
        best_svdd_params = {
            'nu': 0.5,
            'gamma_values': [0.1, 1, 10],  # List format
            'kernel_values': ['linear', 'rbf', 'sigmoid']  # List format
        }
        
        # Enroll the user by training their models
        svdd_model_path, ocknn_model_path = enroll_user(user_id, user_data_path, best_svdd_params)
        
        if svdd_model_path and ocknn_model_path:
            logging.info(f"Models trained and saved for user {user_id}")
            return jsonify({"message": "User profile created and models trained successfully"}), 201
        else:
            logging.error(f"Model training failed for user {user_id}")
            return jsonify({"message": "Failed to train models"}), 500
    
    except KeyError as e:
        logging.error(f"Missing key in request data: {e}")
        return jsonify({"message": f"Missing key: {e}"}), 400
    except Exception as e:
        logging.error(f"Error in create_profile: {e}")
        return jsonify({"message": "An error occurred during profile creation"}), 500


@app.route('/get_profile', methods=['GET'])
def get_profile():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({"message": "Missing user_id parameter"}), 400
        
        # Get the most recent time-stamped folder
        most_recent_folder = max(glob.glob(os.path.join(models_base_directory, '*')), key=os.path.getctime)
        
        # Construct paths to model files
        svdd_model_path = os.path.join(most_recent_folder, f"{user_id}_svdd_model.pkl")
        ocknn_model_path = os.path.join(most_recent_folder, f"{user_id}_ocknn_model.pkl")
        
        if os.path.exists(svdd_model_path) and os.path.exists(ocknn_model_path):
            return jsonify({
                "message": "User models loaded successfully",
                "svdd_model_path": svdd_model_path,
                "ocknn_model_path": ocknn_model_path
            }), 200
        else:
            logging.warning(f"Model files not found for user {user_id}")
            return jsonify({"message": "Model files not found"}), 404
    except Exception as e:
        logging.error(f"Error in get_profile: {e}")
        return jsonify({"message": "An error occurred while loading the profile"}), 500


@app.route('/update_profile', methods=['POST'])
def update_profile():
    try:
        data = request.json
        user_id = data['user_id']
        touch_data = data['touch_data']
        
        # Convert touch_data to DataFrame
        df = pd.DataFrame(touch_data)
        
        # Validate the DataFrame has at least 6 columns
        if df.shape[1] < 6:
            logging.error(f"Touch data for user {user_id} has insufficient columns: {df.shape[1]}")
            return jsonify({"message": "Touch data must have at least 6 columns"}), 400
        
        # Path to the raw touch data CSV
        user_data_path = os.path.join(user_profiles_directory, f"{user_id}_raw.csv")
        
        if os.path.exists(user_data_path):
            existing_df = pd.read_csv(user_data_path, header=None)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            logging.info(f"Appending new touch data to existing data for user {user_id}")
        else:
            updated_df = df
            logging.info(f"Creating new touch data file for user {user_id}")
        
        # Save the updated raw touch data
        updated_df.to_csv(user_data_path, index=False, header=False)
        logging.info(f"Raw touch data updated for user {user_id}")
        
        # Define best SVDD parameters from previous training
        best_svdd_params = {
            'nu': 0.5,
            'gamma_values': [0.01],  # List format
            'kernel_values': ['rbf']  # List format
        }
        
        # Retrain the user model with updated data
        svdd_model_path, ocknn_model_path = enroll_user(user_id, user_data_path, best_svdd_params)
        
        if svdd_model_path and ocknn_model_path:
            logging.info(f"Models retrained and saved for user {user_id}")
            return jsonify({"message": "User profile updated and models retrained successfully"}), 200
        else:
            logging.error(f"Model retraining failed for user {user_id}")
            return jsonify({"message": "Failed to retrain models"}), 500
    
    except KeyError as e:
        logging.error(f"Missing key in request data: {e}")
        return jsonify({"message": f"Missing key: {e}"}), 400
    except Exception as e:
        logging.error(f"Error in update_profile: {e}")
        return jsonify({"message": "An error occurred during profile update"}), 500


if __name__ == "__main__":
    app.run(debug=True)
