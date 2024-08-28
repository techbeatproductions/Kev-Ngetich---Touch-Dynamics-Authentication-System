from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

app = Flask(__name__)

# Directories
BASE_DIR = r'C:\Users\ngeti\Documents\4.2\Final Year Project System'
directory = os.path.join(BASE_DIR, 'touch_dynamics_dataset')
plots_base_directory = os.path.join(BASE_DIR, 'plots')
models_base_directory = os.path.join(BASE_DIR, 'models')

def extract_features(df):
    try:
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)

        tap_timestamps = df.iloc[:, 2]
        tar_timestamps = df.iloc[:, 4]
        pressure_sizes = df.iloc[:, 5]

        dt = np.diff(tap_timestamps, prepend=tap_timestamps.iloc[0])
        ft = tar_timestamps - tap_timestamps
        it = np.diff(tar_timestamps, prepend=tar_timestamps.iloc[0])

        return pd.DataFrame({'PS': pressure_sizes, 'DT': dt, 'FT': ft, 'IT': it})
    except Exception as e:
        print(f"Error in extract_features: {e}")
        return pd.DataFrame()

def generate_illegitimate_samples(X, num_samples):
    try:
        num_samples = min(num_samples, X.shape[0])
        noise_scale = 0.05
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=(num_samples, X.shape[1]))
        return X[np.random.choice(X.shape[0], num_samples, replace=False)] + noise
    except Exception as e:
        print(f"Error in generate_illegitimate_samples: {e}")
        return np.array([])

def load_and_preprocess_data(csv_file_path, k=10):
    try:
        data = pd.read_csv(csv_file_path, header=None)
        data = data[data.iloc[:, 1] != 'E']
        data = data.apply(pd.to_numeric, errors='coerce').dropna()

        features_df = extract_features(data)
        if features_df.empty:
            print(f"No valid data in file '{csv_file_path}'. Skipping...")
            return None, None, None, None
        
        legitimate_labels = np.ones(len(features_df))
        illegitimate_features = generate_illegitimate_samples(features_df.values, 100)  # Increased sample size
        illegitimate_labels = np.full(illegitimate_features.shape[0], -1)

        all_features = np.vstack((features_df.values, illegitimate_features))
        all_labels = np.hstack((legitimate_labels, illegitimate_labels))
        
        scaler = MinMaxScaler()
        X = scaler.fit_transform(all_features)
        
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        X = selector.fit_transform(X, all_labels)
        
        return train_test_split(X, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
    except FileNotFoundError:
        print(f"File '{csv_file_path}' not found. Skipping...")
        return None, None, None, None
    except ValueError as e:
        print(f"ValueError: {e}")
        return None, None, None, None

def train_svdd(X_train, y_train):
    try:
        param_grid = {'kernel': ['rbf', 'linear', 'poly'], 'gamma': [1e-3, 1e-2, 1e-1], 'nu': [0.05, 0.1, 0.2]}
        svdd = OneClassSVM()
        grid_search = GridSearchCV(svdd, param_grid, cv=3, scoring='f1_weighted')  # Changed scoring to F1 Score
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    except Exception as e:
        print(f"Error in train_svdd: {e}")
        return None

def train_ocknn(X_train):
    try:
        # Train LocalOutlierFactor with modified parameters
        lof = LocalOutlierFactor(n_neighbors=5, novelty=True)  # Adjusted n_neighbors
        lof.fit(X_train)
        return lof
    except Exception as e:
        print(f"Error in train_ocknn: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    try:
        if isinstance(model, LocalOutlierFactor):
            # Use the outlier scores for LocalOutlierFactor
            scores = model.decision_function(X_test)
            predictions_binary = np.where(scores < 0, -1, 1)
        else:
            # Use model's predict method for other models
            predictions_binary = model.predict(X_test)
        
        # Handle multiclass classification by setting average to 'weighted'
        precision = precision_score(y_test, predictions_binary, average='weighted', zero_division=1)
        recall = recall_score(y_test, predictions_binary, average='weighted', zero_division=1)
        f1 = f1_score(y_test, predictions_binary, average='weighted', zero_division=1)
        
        cm = confusion_matrix(y_test, predictions_binary)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        far = fp / (fp + tn) if (fp + tn) != 0 else 0
        frr = fn / (fn + tp) if (fn + tp) != 0 else 0

        fpr, tpr, _ = roc_curve(y_test, predictions_binary, pos_label=1)
        eer = np.interp(0.5, np.concatenate([fpr, [1]]), np.concatenate([tpr, [1]]))

        return {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'FAR': far,
            'FRR': frr,
            'EER': eer
        }
    except Exception as e:
        print(f"Error in evaluate_model: {e}")
        return {}

def plot_distributions(svdd_metrics, ocknn_metrics):
    try:
        plot_directory = os.path.join(plots_base_directory, 'metrics')
        os.makedirs(plot_directory, exist_ok=True)

        plt.figure(figsize=(12, 8))
        metrics = ['EER', 'FAR', 'FRR', 'F1 Score']
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            plt.bar(['SVDD', 'OCKNN'], [svdd_metrics.get(metric, 0), ocknn_metrics.get(metric, 0)], color=['blue', 'orange'])
            plt.title(f'{metric} Comparison')
            plt.ylabel(metric)

        plt.tight_layout()
        plot_path = os.path.join(plot_directory, f'metrics_comparison_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Error in plot_distributions: {e}")

@app.route('/enroll', methods=['POST'])
def enroll_user():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.json
        
        user_id = data.get('user_id')
        tap_data = data.get('tap_data')
        
        if not user_id or not tap_data:
            return jsonify({'error': 'Missing user_id or tap_data in request'}), 400
        
        # Convert tap_data to DataFrame
        user_data_df = pd.DataFrame(tap_data, columns=[0, 1, 2, 3, 4, 5])
        
        # Extract features
        user_features_df = extract_features(user_data_df)
        
        if user_features_df.empty:
            return jsonify({'error': 'No valid data in user data'}), 400
        
        user_features = user_features_df.values
        user_labels = np.ones(user_features.shape[0])
        
        # Load and preprocess other CSV files
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        all_features = []
        all_labels = []
        
        for csv_file in csv_files:
            csv_file_path = os.path.join(directory, csv_file)
            X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_file_path)
            if X_train is None:
                continue
            
            all_features.append(X_train)
            all_labels.append(y_train)
        
        if not all_features:
            return jsonify({'error': 'No valid CSV files found for training'}), 400
        
        all_features = np.vstack(all_features)
        all_labels = np.hstack(all_labels)
        
        X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
        
        # Train models
        svdd_model = train_svdd(X_train, y_train)
        ocknn_model = train_ocknn(X_train)
        
        if svdd_model is None or ocknn_model is None:
            return jsonify({'error': 'Error in training models'}), 500
        
        # Evaluate models
        svdd_metrics = evaluate_model(svdd_model, X_test, y_test)
        ocknn_metrics = evaluate_model(ocknn_model, X_test, y_test)
        
        # Save user data and model metrics
        user_model_path = os.path.join(models_base_directory, f'{user_id}_model.pkl')
        with open(user_model_path, 'wb') as model_file:
            pickle.dump({'svdd': svdd_model, 'ocknn': ocknn_model}, model_file)
        
        plot_distributions(svdd_metrics, ocknn_metrics)
        
        return jsonify({
            'message': 'User enrolled successfully',
            'metrics': {
                'SVDD': svdd_metrics,
                'OCKNN': ocknn_metrics
            }
        })
    except Exception as e:
        print(f"Error in enroll_user: {e}")
        return jsonify({'error': 'An error occurred during enrollment'}), 500

if __name__ == '__main__':
    app.run(debug=True)
