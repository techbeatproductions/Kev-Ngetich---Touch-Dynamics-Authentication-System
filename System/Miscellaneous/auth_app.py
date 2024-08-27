from flask import Flask, jsonify, request
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import logging
from model_utils import load_model

app = Flask(__name__)

# Directory where models are saved
models_base_directory = r'C:\Users\ngeti\Documents\4.2\Final Year Project System\models'

logging.basicConfig(level=logging.INFO)

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

def preprocess_data(features_df):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features_df)
    selector = SelectKBest(score_func=mutual_info_classif, k=min(10, X.shape[1]))
    X = selector.fit_transform(X, np.ones(X.shape[0]))  # Dummy labels as we're just transforming
    return X

@app.route('/authenticate', methods=['POST'])
def authenticate():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    df = pd.DataFrame(data)
    features_df = extract_features(df)
    print("Feature dataframe:")
    print(features_df.head())

    if features_df.empty:
        return jsonify({"error": "No valid data to process"}), 400

    X = preprocess_data(features_df)
    print("Scaled features:")
    print(X[:5])

    svdd_model = load_model('svdd')
    ocknn_model = load_model('ocknn')
    logging.info(f"Loaded models: svdd={svdd_model}, ocknn={ocknn_model}")

    if svdd_model is None or ocknn_model is None:
        return jsonify({"error": "Model loading failed"}), 500

    try:
        svdd_predictions = svdd_model.predict(X)
        ocknn_predictions = ocknn_model.predict(X)
        print("SVDD predictions:")
        print(svdd_predictions)
        print("OCKNN predictions:")
        print(ocknn_predictions)
        svdd_predictions_fixed = np.where(svdd_predictions < 0, 0, svdd_predictions)
        ocknn_predictions_fixed = np.where(ocknn_predictions < 0, 0, ocknn_predictions)
        print("SVDD prediction counts:")
        print(np.bincount(svdd_predictions_fixed))
        print("OCKNN prediction counts:")
        print(np.bincount(ocknn_predictions_fixed))
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        return jsonify({"error": "Prediction failed"}), 500

    response = {
        "svdd_authentication": "Legitimate" if np.all(svdd_predictions == 1) else "Illegitimate",
        "ocknn_authentication": "Legitimate" if np.all(ocknn_predictions == 1) else "Illegitimate"
    }

    return jsonify(response)
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    df = pd.DataFrame(data)
    features_df = extract_features(df)
    if features_df.empty:
        return jsonify({"error": "No valid data to process"}), 400

    X = preprocess_data(features_df)

    svdd_model = load_model('svdd')
    ocknn_model = load_model('ocknn')
    logging.info(f"Loaded models: svdd={svdd_model}, ocknn={ocknn_model}")


    if svdd_model is None or ocknn_model is None:
        return jsonify({"error": "Model loading failed"}), 500

    try:
        svdd_predictions = svdd_model.predict(X)
        ocknn_predictions = ocknn_model.predict(X)
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        return jsonify({"error": "Prediction failed"}), 500

    response = {
        "svdd_authentication": "Legitimate" if np.all(svdd_predictions == 1) else "Illegitimate",
        "ocknn_authentication": "Legitimate" if np.all(ocknn_predictions == 1) else "Illegitimate"
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)