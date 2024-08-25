from flask import Flask, jsonify, request
import logging
import os
import joblib
import numpy as np
import pandas as pd
from train_and_save_models import train_user_model  # Import your function directly

app = Flask(__name__)

# Get the base directory where user profiles are saved
user_profiles_directory = r'C:\Users\ngeti\Documents\4.2\Final Year Project System\user_profiles'

def extract_features(df):
    df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')  # TAP timestamp
    df.iloc[:, 4] = pd.to_numeric(df.iloc[:, 4], errors='coerce')  # TAR timestamp
    df.iloc[:, 5] = pd.to_numeric(df.iloc[:, 5], errors='coerce')  # TAP pressure size
    
    df = df.dropna()

    # Extract columns for feature computation
    tap_timestamps = df.iloc[:, 2].values
    tar_timestamps = df.iloc[:, 4].values
    pressure_sizes = df.iloc[:, 5].values
    
    # Calculate DT, FT, and IT
    dt = np.diff(tap_timestamps, prepend=tap_timestamps[0])
    ft = tar_timestamps - tap_timestamps
    it = np.diff(tar_timestamps, prepend=tar_timestamps[0])
    
    # Create features DataFrame
    features_df = pd.DataFrame({
        'PS': pressure_sizes,
        'DT': dt,
        'FT': ft,
        'IT': it
    })
    
    return features_df

def save_user_profile(user_id, user_profile):
    if not os.path.exists(user_profiles_directory):
        os.makedirs(user_profiles_directory)
    
    user_profile_path = os.path.join(user_profiles_directory, f"{user_id}.pkl")
    joblib.dump(user_profile, user_profile_path)
    logging.info(f"User profile saved to: {user_profile_path}")

@app.route('/create_profile', methods=['POST'])
def create_profile():
    data = request.json
    user_id = data['user_id']
    touch_data = data['touch_data']
    
    df = pd.DataFrame(touch_data)
    features_df = extract_features(df)
    
    if features_df.empty:
        return jsonify({"message": "Failed to create user profile"}), 500
    
    # Save the features as a CSV file for the user
    user_data_path = os.path.join(user_profiles_directory, f"{user_id}.csv")
    features_df.to_csv(user_data_path, index=False)
    
    # Assume that best_svdd_params is a dictionary with the best SVDD parameters from your previous training.
    best_svdd_params = {
        'nu': 0.5,
        'gamma_values': [0.001, 0.01, 0.1],
        'kernel_values': ['rbf']
    }
    
    # Train the user model
    svdd_model_path, ocknn_model_path = train_user_model(user_id, user_data_path, best_svdd_params)
    
    if svdd_model_path and ocknn_model_path:
        return jsonify({"message": "User profile created and models trained successfully"}), 201
    else:
        return jsonify({"message": "Failed to train models"}), 500

    data = request.json
    user_id = data['user_id']
    touch_data = data['touch_data']
    
    df = pd.DataFrame(touch_data)
    features_df = extract_features(df)
    
    if features_df.empty:
        return jsonify({"message": "Failed to create user profile"}), 500
    
    # Use the extracted features for training
    user_data = features_df.values
    
    # Assume that best_svdd_metrics is a dictionary with the best SVDD parameters from your previous training.
    best_svdd_metrics = {
        'nu': 0.5,
        'gamma_values': [0.001, 0.01, 0.1],
        'kernel_values': ['rbf']
    }
    
    # Train the user model
    svdd_model_path, ocknn_model_path = train_user_model(user_id, user_data, best_svdd_metrics)
    
    # Save the trained user profile
    save_user_profile(user_id, user_data)
    
    return jsonify({"message": "User profile created and models trained successfully"}), 201

if __name__ == "__main__":
    app.run(debug=True)
