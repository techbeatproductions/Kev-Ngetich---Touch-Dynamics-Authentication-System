from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
from joblib import Parallel, delayed

app = Flask(__name__)

# Directory containing the CSV files and plots
directory = r'C:\Users\ngeti\Documents\4.2\Final Year Project System\touch_dynamics_dataset/'
plots_base_directory = r'C:\Users\ngeti\Documents\4.2\Final Year Project System\plots/'

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

def generate_illegitimate_samples(X, num_samples):
    num_samples = min(num_samples, X.shape[0])
    noise_scale = 0.05  # Adjust this scale based on your data
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=X.shape)
    illegitimate_samples = X + noise
    indices = np.random.choice(X.shape[0], num_samples, replace=False)
    return illegitimate_samples[indices]

def load_and_preprocess_data(csv_file_path, k=5, num_illegitimate_samples=30):
    try:
        data = pd.read_csv(csv_file_path, header=None)
        data = data[data.iloc[:, 1] != 'E']
        data = data.dropna()
        features_df = extract_features(data)
        
        if features_df.empty:
            print(f"No valid data in file '{csv_file_path}'. Skipping...")
            return None, None, None, None
        
        legitimate_labels = np.ones(len(features_df))
        
        if len(features_df) > 0:
            illegitimate_features = generate_illegitimate_samples(features_df.values, num_illegitimate_samples)
            illegitimate_labels = np.ones(illegitimate_features.shape[0]) * -1
            
            all_features = np.vstack((features_df.values, illegitimate_features))
            all_labels = np.hstack((legitimate_labels, illegitimate_labels))
        else:
            all_features = features_df.values
            all_labels = legitimate_labels
        
        y_binary = np.where(all_labels == 1, 1, -1)
        
        scaler = MinMaxScaler()
        X = scaler.fit_transform(all_features)
        
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        X = selector.fit_transform(X, y_binary)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, stratify=y_binary, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    except FileNotFoundError:
        print(f"File '{csv_file_path}' not found. Skipping...")
        return None, None, None, None
    
    except ValueError as e:
        print(f"ValueError: {e}")
        return None, None, None, None

def train_svdd(X_train, gamma=0.05, kernel='rbf', nu=0.1):
    if X_train is None:
        return None
    
    svdd = OneClassSVM(gamma=gamma, kernel=kernel, nu=nu)
    svdd.fit(X_train)
    return svdd

def train_ocknn(X_train, n_neighbors=10):
    ocknn = LocalOutlierFactor(novelty=True, n_neighbors=n_neighbors)
    ocknn.fit(X_train)
    return ocknn

def train_isolation_forest(X_train, n_estimators=200, contamination='auto'):
    if X_train is None:
        return None
    
    isolation_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    isolation_forest.fit(X_train)
    return isolation_forest

def evaluate_model(model, X_test, y_test, timestamp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        predictions = model.predict(X_test)
        predictions_binary = np.where(predictions == 1, 1, 0)

        y_test_binary = np.where(y_test == 1, 1, 0)

        # Calculate precision, recall, F1 score
        precision = precision_score(y_test_binary, predictions_binary, zero_division=1)
        recall = recall_score(y_test_binary, predictions_binary, zero_division=1)
        f1 = f1_score(y_test_binary, predictions_binary, zero_division=1)

        # Confusion matrix
        cm = confusion_matrix(y_test_binary, predictions_binary)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        # Calculate Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

        # FAR and FRR
        far = fp / (fp + tn) if (fp + tn) != 0 else 0
        frr = fn / (fn + tp) if (fn + tp) != 0 else 0

        # ROC Curve and EER
        try:
            fpr, tpr, _ = roc_curve(y_test_binary, predictions_binary, pos_label=1)
            eer = np.interp(0.5, np.concatenate([fpr, [1]]), np.concatenate([tpr, [1]]))
        except ValueError:
            eer = np.nan

        metrics = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Accuracy': accuracy,  # Report accuracy as a value between 0 and 1
            'FAR': far,
            'FRR': frr,
            'EER': eer
        }

        # Plot the confusion matrix
        plot_directory = os.path.join(plots_base_directory, timestamp)
        os.makedirs(plot_directory, exist_ok=True)

        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(plot_directory, 'confusion_matrix.png'))
        plt.close()

        # Precision-Recall Curve
        plt.figure(figsize=(10, 5))
        precision_vals, recall_vals, _ = precision_recall_curve(y_test_binary, predictions_binary, pos_label=1)
        plt.plot(recall_vals, precision_vals, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(os.path.join(plot_directory, 'precision_recall_curve.png'))
        plt.close()

        # ROC Curve
        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(os.path.join(plot_directory, 'roc_curve.png'))
        plt.close()

        return metrics


def plot_distributions(svdd_metrics, ocknn_metrics, isolation_forest_metrics, timestamp):
    # Create a timestamped folder in the plots directory
    plot_directory = os.path.join(plots_base_directory, timestamp)
    os.makedirs(plot_directory, exist_ok=True)

    metric_names = list(svdd_metrics.keys())
    
    # Plot and save SVDD metrics
    metric_values = [value for value in svdd_metrics.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(metric_names)), metric_values)
    plt.xticks(range(len(metric_names)), metric_names)
    plt.title('SVDD Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.savefig(os.path.join(plot_directory, 'svdd_metrics.png'))
    plt.close()

    # Plot and save OCKNN metrics
    metric_values = [value for value in ocknn_metrics.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(metric_names)), metric_values)
    plt.xticks(range(len(metric_names)), metric_names)
    plt.title('OCKNN Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.savefig(os.path.join(plot_directory, 'ocknn_metrics.png'))
    plt.close()

    # Plot and save Isolation Forest metrics
    metric_values = [value for value in isolation_forest_metrics.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(metric_names)), metric_values)
    plt.xticks(range(len(metric_names)), metric_names)
    plt.title('Isolation Forest Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.savefig(os.path.join(plot_directory, 'isolation_forest_metrics.png'))
    plt.close()

@app.route('/enroll_user', methods=['POST'])
def enroll_user():
    user_id = request.json['user_id']  # This is still received, but not used for file filtering
    tap_data = request.json['tap_data']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    if not csv_files:
        return jsonify({"error": "No CSV files found in the directory."}), 404

    # Process all CSV files
    results = Parallel(n_jobs=os.cpu_count())(
        delayed(load_and_preprocess_data)(os.path.join(directory, csv_file), k=5, num_illegitimate_samples=10)
        for csv_file in csv_files
    )

    # Filter out None values and concatenate the results
    X_train_list, X_test_list, y_train_list, y_test_list = zip(*results)
    X_train = np.vstack([X for X in X_train_list if X is not None])
    X_test = np.vstack([X for X in X_test_list if X is not None])
    y_train = np.hstack([y for y in y_train_list if y is not None])
    y_test = np.hstack([y for y in y_test_list if y is not None])

    if X_train is None or X_test is None:
        return jsonify({"error": "No valid training data available."}), 500

    svdd_model = train_svdd(X_train)
    ocknn_model = train_ocknn(X_train)
    isolation_forest_model = train_isolation_forest(X_train)

    svdd_metrics = evaluate_model(svdd_model, X_test, y_test, timestamp)
    ocknn_metrics = evaluate_model(ocknn_model, X_test, y_test, timestamp)
    isolation_forest_metrics = evaluate_model(isolation_forest_model, X_test, y_test, timestamp)

    plot_distributions(svdd_metrics, ocknn_metrics, isolation_forest_metrics, timestamp)

    # Save models
    model_path = os.path.join(plots_base_directory, timestamp)
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, 'svdd_model.pkl'), 'wb') as f:
        pickle.dump(svdd_model, f)
    with open(os.path.join(model_path, 'ocknn_model.pkl'), 'wb') as f:
        pickle.dump(ocknn_model, f)
    with open(os.path.join(model_path, 'isolation_forest_model.pkl'), 'wb') as f:
        pickle.dump(isolation_forest_model, f)

    return jsonify({
        "user_id": user_id,
        "timestamp": timestamp,
        "svdd_metrics": svdd_metrics,
        "ocknn_metrics": ocknn_metrics,
        "isolation_forest_metrics": isolation_forest_metrics
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
