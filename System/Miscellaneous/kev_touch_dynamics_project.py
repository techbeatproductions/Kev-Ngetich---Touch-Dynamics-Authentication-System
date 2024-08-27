from flask import Flask, jsonify, request, render_template
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import datetime

app = Flask(__name__)

# Directory containing the CSV files
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

def load_and_preprocess_data(csv_file_path, k=10, num_illegitimate_samples=40, verbose=False):
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
            illegitimate_labels = np.ones(num_illegitimate_samples) * -1
            
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

def train_svdd(X_train, y_train):
    if X_train is None or y_train is None:
        return None
    
    param_grid = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                  'gamma': [1e-3, 1e-2, 0.1, 1, 10, 100],
                  'nu': [1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9]}
    svdd = OneClassSVM()
    grid_search = GridSearchCV(svdd, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_ocknn(X_train):
    ocknn = LocalOutlierFactor(novelty=True)
    ocknn.fit(X_train)
    return ocknn

def evaluate_model(model, X_test, y_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if isinstance(model, LocalOutlierFactor):
            predictions = model.predict(X_test)
            predictions_binary = np.where(predictions == 1, 1, 0)
        else:
            predictions = model.predict(X_test)
            predictions_binary = np.where(predictions == 1, 1, 0)

        y_test_binary = np.where(y_test == 1, 1, 0)

        precision = precision_score(y_test_binary, predictions_binary, zero_division=1)
        recall = recall_score(y_test_binary, predictions_binary, zero_division=1)
        f1 = f1_score(y_test_binary, predictions_binary, zero_division=1)

        cm = confusion_matrix(y_test_binary, predictions_binary)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        far = fp / (fp + tn) if (fp + tn) != 0 else 0
        frr = fn / (fn + tp) if (fn + tp) != 0 else 0

        try:
            fpr, tpr, thresholds = roc_curve(y_test_binary, predictions_binary, pos_label=1)
            eer = np.interp(0.5, np.concatenate([fpr, [1]]), np.concatenate([tpr, [1]]))
        except ValueError:
            eer = np.nan

        return precision, recall, f1, far, frr, cm, eer

def plot_distributions(metrics, model_name, run_directory):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(metrics['far'], bins=20, density=True, color='blue', alpha=0.7)
    plt.title(f'{model_name} FAR Distribution')
    plt.xlabel('FAR')
    plt.ylabel('Density')

    plt.subplot(1, 3, 2)
    plt.hist(metrics['frr'], bins=20, density=True, color='red', alpha=0.7)
    plt.title(f'{model_name} FRR Distribution')
    plt.xlabel('FRR')
    plt.ylabel('Density')

    plt.subplot(1, 3, 3)
    plt.hist(metrics['eer'], bins=20, density=True, color='green', alpha=0.7)
    plt.title(f'{model_name} EER Distribution')
    plt.xlabel('EER')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.savefig(os.path.join(run_directory, f'{model_name}_distributions.png'))
    plt.close()

def plot_confusion_matrix(cm, model_name, run_directory):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(run_directory, f'{model_name}_confusion_matrix.png'))
    plt.close()

def main():
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    num_files = len(csv_files)
    
    if num_files == 0:
        print("No CSV files found in the directory.")
        return

    print(f"Processing {num_files} files...")

    metrics = {
        'far': [],
        'frr': [],
        'eer': []
    }
    
    confusion_matrices = {
        'SVDD': [],
        'OCKNN': []
    }

    # Create a directory with a timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = os.path.join(plots_base_directory, f'run_{timestamp}')
    os.makedirs(run_directory, exist_ok=True)

    for i, file_name in enumerate(tqdm(csv_files, desc="Processing files", unit="file")):
        file_path = os.path.join(directory, file_name)
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, verbose=False)
        
        if X_train is None or y_train is None:
            continue
        
        svdd_model = train_svdd(X_train, y_train)
        ocknn_model = train_ocknn(X_train)

        if svdd_model:
            precision, recall, f1, far, frr, cm, eer = evaluate_model(svdd_model, X_test, y_test)
            metrics['far'].append(far)
            metrics['frr'].append(frr)
            metrics['eer'].append(eer)
            confusion_matrices['SVDD'].append(cm)

        if ocknn_model:
            precision, recall, f1, far, frr, cm, eer = evaluate_model(ocknn_model, X_test, y_test)
            metrics['far'].append(far)
            metrics['frr'].append(frr)
            metrics['eer'].append(eer)
            confusion_matrices['OCKNN'].append(cm)
    
    # Plot and save distributions for SVDD
    plot_distributions(metrics, 'SVDD', run_directory)
    # Plot and save distributions for OCKNN
    plot_distributions(metrics, 'OCKNN', run_directory)

    # Save confusion matrices for SVDD and OCKNN
    if confusion_matrices['SVDD']:
        avg_cm_svdd = np.mean(confusion_matrices['SVDD'], axis=0).astype(int)
        plot_confusion_matrix(avg_cm_svdd, 'SVDD', run_directory)

    if confusion_matrices['OCKNN']:
        avg_cm_ocknn = np.mean(confusion_matrices['OCKNN'], axis=0).astype(int)
        plot_confusion_matrix(avg_cm_ocknn, 'OCKNN', run_directory)

    print(f"Processing completed. Plots saved in {run_directory}")

if __name__ == '__main__':
    main()
