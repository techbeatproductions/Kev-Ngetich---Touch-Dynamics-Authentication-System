from flask import Flask, jsonify, request
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import datetime

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

def load_and_preprocess_data(csv_file_path, k=10, num_illegitimate_samples=20, verbose=False):
    try:
        data = pd.read_csv(csv_file_path, header=None)
        data = data[data.iloc[:, 1] != 'E']
        data.iloc[:, 2] = pd.to_numeric(data.iloc[:, 2], errors='coerce')  # TAP timestamp
        data.iloc[:, 4] = pd.to_numeric(data.iloc[:, 4], errors='coerce')  # TAR timestamp
        data.iloc[:, 5] = pd.to_numeric(data.iloc[:, 5], errors='coerce')  # TAP pressure size
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
            
            if verbose:
                print(f"Legitimate features shape: {features_df.shape}")
                print(f"Illegitimate features shape: {illegitimate_features.shape}")
                print(f"All features shape: {all_features.shape}")
                print(f"All labels shape: {all_labels.shape}")
            
            if len(all_labels) != len(all_features):
                print(f"Mismatch detected: len(all_features) = {len(all_features)}, len(all_labels) = {len(all_labels)}")
                raise ValueError("Mismatch between number of samples in features and labels")
        else:
            all_features = features_df.values
            all_labels = legitimate_labels
        
        y_binary = np.where(all_labels == 1, 1, -1)
        
        scaler = MinMaxScaler()
        X = scaler.fit_transform(all_features)
        
        if verbose:
            print(f"Features shape after scaling: {X.shape}")
            print(f"Labels shape before feature selection: {y_binary.shape}")
        
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        X = selector.fit_transform(X, y_binary)
        
        if X.shape[0] != len(y_binary):
            print(f"Mismatch detected after feature selection: X.shape[0] = {X.shape[0]}, len(y_binary) = {len(y_binary)}")
            raise ValueError("Mismatch between number of samples after feature selection")
        
        if verbose:
            print(f"Features shape after feature selection: {X.shape}")
            print(f"Labels shape after feature selection: {y_binary.shape}")
        
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

def train_svdd(X_train, y_train, gamma_values=[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000], kernel_values=['rbf', 'linear', 'poly', 'sigmoid'], nu_values=[1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 0.95]):
    if X_train is None or y_train is None:
        return None
    
    param_grid = {'kernel': kernel_values, 'gamma': gamma_values, 'nu': nu_values}
    svdd = OneClassSVM()
    grid_search = GridSearchCV(svdd, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_ocknn(X_train):
    ocknn = LocalOutlierFactor(novelty=True)
    ocknn.fit(X_train)
    return ocknn

def evaluate_model(model, X_test, y_test, timestamp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if isinstance(model, LocalOutlierFactor):
            predictions = model.predict(X_test)
            predictions_binary = np.where(predictions == 1, 1, 0)
        else:
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
        accuracy_percentage = accuracy * 100

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
            'Accuracy': accuracy_percentage,  # Include accuracy as a percentage
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




def plot_distributions(svdd_metrics, ocknn_metrics, timestamp):
    # Create a timestamped folder in the plots directory
    plot_directory = os.path.join(plots_base_directory, timestamp)
    os.makedirs(plot_directory, exist_ok=True)

    # Plot and save SVDD metrics
    metric_names = list(svdd_metrics.keys())
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
    metric_names = list(ocknn_metrics.keys())
    metric_values = [value for value in ocknn_metrics.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(metric_names)), metric_values)
    plt.xticks(range(len(metric_names)), metric_names)
    plt.title('OCKNN Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.savefig(os.path.join(plot_directory, 'ocknn_metrics.png'))
    plt.close()

@app.route('/enroll', methods=['POST'])
def enroll_user():
    try:
        # Load and preprocess data from the POST request
        request_data = request.get_json()  # Get the JSON data from the request
        tap_data = request_data.get('tap_data')
        user_id = request_data.get('user_id')
        
        if tap_data is None or user_id is None:
            return jsonify({'error': 'No user data or user ID provided'}), 400

        # Convert the list of lists into a DataFrame
        user_df = pd.DataFrame(tap_data, columns=['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6'])
        user_features_df = extract_features(user_df)
        
        if user_features_df.empty:
            return jsonify({'error': 'No valid data in user request'}), 400
        
        legitimate_user_labels = np.ones(len(user_features_df))
        
        # Load and preprocess existing dataset
        existing_features_list = []
        existing_labels_list = []

        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        for csv_file in csv_files:
            csv_file_path = os.path.join(directory, csv_file)
            print(f"Processing file: {csv_file_path}")
            
            try:
                X_train, _, y_train, _ = load_and_preprocess_data(csv_file_path, k=10, num_illegitimate_samples=40, verbose=False)
                
                if X_train is not None and y_train is not None:
                    legitimate_existing_labels = np.ones(len(y_train))
                    
                    existing_features_list.append(X_train)
                    existing_labels_list.append(legitimate_existing_labels)
                    
            except Exception as e:
                print(f"Error processing file {csv_file_path}: {e}")

        if not existing_features_list:
            return jsonify({'error': 'No valid data in existing dataset'}), 400
        
        # Combine user features with existing features
        existing_features = np.vstack(existing_features_list)
        existing_labels = np.hstack(existing_labels_list)
        
        illegitimate_features = generate_illegitimate_samples(existing_features, num_samples=len(existing_features))
        illegitimate_labels = np.ones(len(illegitimate_features)) * -1
        
        all_features = np.vstack((existing_features, illegitimate_features, user_features_df.values))
        all_labels = np.hstack((existing_labels, illegitimate_labels, legitimate_user_labels))

        # Preprocess combined data
        scaler = MinMaxScaler()
        X = scaler.fit_transform(all_features)
        y_binary = np.where(all_labels == 1, 1, -1)
        
        selector = SelectKBest(score_func=mutual_info_classif, k=min(10, X.shape[1]))
        X = selector.fit_transform(X, y_binary)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, stratify=y_binary, random_state=42
        )
        
        # Train models
        print("Training SVDD model...")
        best_svdd = train_svdd(X_train, y_train)
        if best_svdd is None:
            return jsonify({'error': 'SVDD model training failed'}), 500

        print("Training OCKNN model...")
        ocknn_model = train_ocknn(X_train)
        if ocknn_model is None:
            return jsonify({'error': 'OCKNN model training failed'}), 500
        
        # Evaluate models
        print("Evaluating SVDD model...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        svdd_metrics = evaluate_model(best_svdd, X_test, y_test, timestamp)
        print("Evaluating OCKNN model...")
        ocknn_metrics = evaluate_model(ocknn_model, X_test, y_test, timestamp)

        # Save models
        user_model_directory = os.path.join(r'C:\Users\ngeti\Documents\4.2\Final Year Project System\models', f'user_{user_id}')
        os.makedirs(user_model_directory, exist_ok=True)
        
        with open(os.path.join(user_model_directory, 'svdd_model.pkl'), 'wb') as f:
            pickle.dump(best_svdd, f)
        
        with open(os.path.join(user_model_directory, 'ocknn_model.pkl'), 'wb') as f:
            pickle.dump(ocknn_model, f)

        # Plot and save metrics
        plot_distributions({
            'FAR': svdd_metrics['FAR'],
            'FRR': svdd_metrics['FRR'],
            'EER': svdd_metrics['EER'],
        }, {
            'FAR': ocknn_metrics['FAR'],
            'FRR': ocknn_metrics['FRR'],
            'EER': ocknn_metrics['EER'],
        }, timestamp)

        return jsonify({
            'message': 'User enrolled successfully',
            'svdd_metrics': svdd_metrics,
            'ocknn_metrics': ocknn_metrics,
            'user_model_directory': user_model_directory,
            'timestamp': timestamp
        })
    
    except Exception as e:
        print(f"Error during enrollment: {e}")
        return jsonify({'error': str(e)}), 500



    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        csv_file_path = os.path.join(directory, csv_file)
        print(f"Processing file: {csv_file_path}")

        try:
            X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_file_path, k=10, num_illegitimate_samples=40, verbose=True)

            if X_train is None or X_test is None or y_train is None or y_test is None:
                print(f"Skipping file due to preprocessing issues: {csv_file_path}")
                continue

            print("Training SVDD model...")
            best_svdd = train_svdd(X_train, y_train)
            if best_svdd is None:
                print(f"SVDD model training failed for file: {csv_file_path}")
                continue

            print("Training OCKNN model...")
            ocknn_model = train_ocknn(X_train)
            if ocknn_model is None:
                print(f"OCKNN model training failed for file: {csv_file_path}")
                continue

            print("Evaluating SVDD model...")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            svdd_metrics = evaluate_model(best_svdd, X_test, y_test, timestamp)
            print("Evaluating OCKNN model...")
            ocknn_metrics = evaluate_model(ocknn_model, X_test, y_test, timestamp)

            # Save models
            user_model_directory = os.path.join(r'C:\Users\ngeti\Documents\4.2\Final Year Project System\models', f'file_{os.path.splitext(csv_file)[0]}')
            os.makedirs(user_model_directory, exist_ok=True)

            with open(os.path.join(user_model_directory, 'svdd_model.pkl'), 'wb') as f:
                pickle.dump(best_svdd, f)

            with open(os.path.join(user_model_directory, 'ocknn_model.pkl'), 'wb') as f:
                pickle.dump(ocknn_model, f)

            # Plot and save metrics
            plot_distributions({
                'FAR': svdd_metrics['FAR'],
                'FRR': svdd_metrics['FRR'],
                'EER': svdd_metrics['EER'],
            }, {
                'FAR': ocknn_metrics['FAR'],
                'FRR': ocknn_metrics['FRR'],
                'EER': ocknn_metrics['EER'],
            }, timestamp)

        except Exception as e:
            print(f"Error processing file {csv_file_path}: {e}")

@app.route('/authenticate', methods=['POST'])
def authenticate_user():

    try:
        # Load the user's tap data and user ID from the request
        request_data = request.get_json()
        tap_data = request_data.get('tap_data')
        user_id = request_data.get('user_id')

        if tap_data is None or user_id is None:
            return jsonify({'error': 'No user data or user ID provided'}), 400

        # Convert the list of lists into a DataFrame
        user_df = pd.DataFrame(tap_data, columns=['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6'])

        # Load the user's model from the file using the user ID
        user_model_directory = os.path.join(r'C:\Users\ngeti\Documents\4.2\Final Year Project System\models', f'user_{user_id}')
        if not os.path.exists(user_model_directory):
            return jsonify({'error': 'User ID not found'}), 404

        svdd_model_path = os.path.join(user_model_directory, 'svdd_model.pkl')
        ocknn_model_path = os.path.join(user_model_directory, 'ocknn_model.pkl')

        if not os.path.exists(svdd_model_path) or not os.path.exists(ocknn_model_path):
            return jsonify({'error': 'User model not found'}), 404

        with open(svdd_model_path, 'rb') as f:
            svdd_model = pickle.load(f)

        with open(ocknn_model_path, 'rb') as f:
            ocknn_model = pickle.load(f)

        # Preprocess the user's tap data
        X = extract_features(user_df)

        # Use the trained models to verify the user's identity
        svdd_prediction = svdd_model.predict(X)
        ocknn_prediction = ocknn_model.predict(X)

        # Check if all predictions are True
        if np.all(svdd_prediction == -1) and np.all(ocknn_prediction == -1):
            return jsonify({'authenticated': True})
        else:
            return jsonify({'authenticated': False})

    except Exception as e:
        print(f"Error during authentication: {e}")
        return jsonify({'error': str(e)}), 500


    try:
        process_all_csv_files()
        return jsonify({'message': 'Processing all CSV files completed successfully'}), 200
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/process_all_csv', methods=['GET'])
def process_all_csv():
    try:
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        svdd_metrics_list = []
        ocknn_metrics_list = []

        for csv_file in csv_files:
            csv_file_path = os.path.join(directory, csv_file)
            print(f"Processing file: {csv_file_path}")

            try:
                X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_file_path, k=10, num_illegitimate_samples=40, verbose=True)

                if X_train is None or X_test is None or y_train is None or y_test is None:
                    print(f"Skipping file due to preprocessing issues: {csv_file_path}")
                    continue

                print("Training SVDD model...")
                best_svdd = train_svdd(X_train, y_train)
                if best_svdd is None:
                    print(f"SVDD model training failed for file: {csv_file_path}")
                    continue

                print("Training OCKNN model...")
                ocknn_model = train_ocknn(X_train)
                if ocknn_model is None:
                    print(f"OCKNN model training failed for file: {csv_file_path}")
                    continue

                print("Evaluating SVDD model...")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                svdd_metrics = evaluate_model(best_svdd, X_test, y_test, timestamp)
                print("Evaluating OCKNN model...")
                ocknn_metrics = evaluate_model(ocknn_model, X_test, y_test, timestamp)

                svdd_metrics_list.append(svdd_metrics)
                ocknn_metrics_list.append(ocknn_metrics)

                # Save models
                user_model_directory = os.path.join(r'C:\Users\ngeti\Documents\4.2\Final Year Project System\models', f'file_{os.path.splitext(csv_file)[0]}')
                os.makedirs(user_model_directory, exist_ok=True)

                with open(os.path.join(user_model_directory, 'svdd_model.pkl'), 'wb') as f:
                    pickle.dump(best_svdd, f)

                with open(os.path.join(user_model_directory, 'ocknn_model.pkl'), 'wb') as f:
                    pickle.dump(ocknn_model, f)

                # Plot and save metrics
                plot_distributions({
                    'FAR': svdd_metrics['FAR'],
                    'FRR': svdd_metrics['FRR'],
                    'EER': svdd_metrics['EER'],
                }, {
                    'FAR': ocknn_metrics['FAR'],
                    'FRR': ocknn_metrics['FRR'],
                    'EER': ocknn_metrics['EER'],
                }, timestamp)

            except Exception as e:
                print(f"Error processing file {csv_file_path}: {e}")

        # Calculate average metrics
        avg_svdd_metrics = {
            'Precision': np.mean([metrics['Precision'] for metrics in svdd_metrics_list]),
            'Recall': np.mean([metrics['Recall'] for metrics in svdd_metrics_list]),
            'F1 Score': np.mean([metrics['F1 Score'] for metrics in svdd_metrics_list]),
            'FAR': np.mean([metrics['FAR'] for metrics in svdd_metrics_list]),
            'FRR': np.mean([metrics['FRR'] for metrics in svdd_metrics_list]),
            'EER': np.mean([metrics['EER'] for metrics in svdd_metrics_list]),
        }

        avg_ocknn_metrics = {
            'Precision': np.mean([metrics['Precision'] for metrics in ocknn_metrics_list]),
            'Recall': np.mean([metrics['Recall'] for metrics in ocknn_metrics_list]),
            'F1 Score': np.mean([metrics['F1 Score'] for metrics in ocknn_metrics_list]),
            'FAR': np.mean([metrics['FAR'] for metrics in ocknn_metrics_list]),
            'FRR': np.mean([metrics['FRR'] for metrics in ocknn_metrics_list]),
            'EER': np.mean([metrics['EER'] for metrics in ocknn_metrics_list]),
        }

        # Save average metrics plots
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_distributions(avg_svdd_metrics, avg_ocknn_metrics, timestamp)

        return jsonify({
            'message': 'Processing all CSV files completed successfully',
            'average_svdd_metrics': avg_svdd_metrics,
            'average_ocknn_metrics': avg_ocknn_metrics,
            'timestamp': timestamp
        }), 200
    
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({'error': str(e)}), 500

    
if __name__ == '__main__':
    app.run(debug=True)
