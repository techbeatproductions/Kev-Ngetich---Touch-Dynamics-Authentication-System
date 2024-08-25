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
import joblib
import datetime

# Directories
directory = r'C:\Users\ngeti\Documents\4.2\Final Year Project System\touch_dynamics_dataset/'
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
plots_base_directory = os.path.join(r'C:\Users\ngeti\Documents\4.2\Final Year Project System\plots', timestamp)
models_base_directory = os.path.join(r'C:\Users\ngeti\Documents\4.2\Final Year Project System\models', timestamp)

# Create directories if they don't exist
os.makedirs(plots_base_directory, exist_ok=True)
os.makedirs(models_base_directory, exist_ok=True)

def extract_features(df):
    df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')  # TAP timestamp
    df.iloc[:, 4] = pd.to_numeric(df.iloc[:, 4], errors='coerce')  # TAR timestamp
    df.iloc[:, 5] = pd.to_numeric(df.iloc[:, 5], errors='coerce')  # TAP pressure size
    df = df.dropna()

    tap_timestamps = df.iloc[:, 2].values
    tar_timestamps = df.iloc[:, 4].values
    pressure_sizes = df.iloc[:, 5].values
    
    dt = np.diff(tap_timestamps, prepend=tap_timestamps[0])
    ft = tar_timestamps - tap_timestamps
    it = np.diff(tar_timestamps, prepend=tar_timestamps[0])
    
    features_df = pd.DataFrame({
        'PS': pressure_sizes,
        'DT': dt,
        'FT': ft,
        'IT': it
    })
    
    return features_df

def generate_illegitimate_samples(X, num_samples, noise_scale):
    num_samples = min(num_samples, X.shape[0])
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=X.shape)
    illegitimate_samples = X + noise
    indices = np.random.choice(X.shape[0], num_samples, replace=False)
    return illegitimate_samples[indices]

def load_and_preprocess_data(csv_file_path, k=10, num_illegitimate_samples=40, noise_scale=0.05):
    try:
        data = pd.read_csv(csv_file_path, header=None)
        data = data[data.iloc[:, 1] != 'E']
        data.iloc[:, 2] = pd.to_numeric(data.iloc[:, 2], errors='coerce')
        data.iloc[:, 4] = pd.to_numeric(data.iloc[:, 4], errors='coerce')
        data.iloc[:, 5] = pd.to_numeric(data.iloc[:, 5], errors='coerce')
        data = data.dropna()
        features_df = extract_features(data)
        
        if features_df.empty:
            print(f"No valid data in file '{csv_file_path}'. Skipping...")
            return None, None, None, None
        
        legitimate_labels = np.ones(len(features_df))
        illegitimate_features = generate_illegitimate_samples(features_df.values, num_illegitimate_samples, noise_scale)
        illegitimate_labels = np.ones(num_illegitimate_samples) * -1
        
        all_features = np.vstack((features_df.values, illegitimate_features))
        all_labels = np.hstack((legitimate_labels, illegitimate_labels))
        
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

def train_svdd(X_train, y_train, nu, gamma_values, kernel_values):
    if X_train is None or y_train is None:
        return None

    param_grid = {
        'kernel': kernel_values if isinstance(kernel_values, list) else [kernel_values],
        'gamma': gamma_values if isinstance(gamma_values, list) else [gamma_values],
        'nu': [nu]
    }

    svdd = OneClassSVM()
    grid_search = GridSearchCV(svdd, param_grid, cv=3, scoring='f1')
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

def plot_distributions(metrics, model_name):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(metrics['far'], bins=20, density=True, color='blue', alpha=0.7)
    plt.title(f'{model_name} FAR Distribution')
    plt.xlabel('FAR')
    plt.ylabel('Density')

    plt.subplot(1, 3, 2)
    plt.hist(metrics['frr'], bins=20, density=True, color='green', alpha=0.7)
    plt.title(f'{model_name} FRR Distribution')
    plt.xlabel('FRR')
    plt.ylabel('Density')

    plt.subplot(1, 3, 3)
    plt.hist(metrics['eer'], bins=20, density=True, color='red', alpha=0.7)
    plt.title(f'{model_name} EER Distribution')
    plt.xlabel('EER')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_base_directory, f'{model_name}_distributions.png'))
    plt.close()

def save_model(model, model_name):
    model_path = os.path.join(models_base_directory, f'{model_name}.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def main():
    # Define parameter values
    nu_values = [0.1, 0.5, 0.9]
    gamma_values = [0.1, 1, 10]
    kernel_values = ['linear', 'rbf', 'sigmoid']
    k_values = [5, 10, 20]
    num_illegitimate_samples_values = [20, 40, 60]
    noise_scales = [0.01, 0.05, 0.1]
    
    metrics_results_svdd = []
    metrics_results_ocknn = []

    for csv_file in os.listdir(directory):
        if csv_file.endswith('.csv'):
            print(f"Processing file: {csv_file}")
            csv_file_path = os.path.join(directory, csv_file)
            for k in k_values:
                for num_illegitimate_samples in num_illegitimate_samples_values:
                    for noise_scale in noise_scales:
                        X_train, X_test, y_train, y_test = load_and_preprocess_data(
                            csv_file_path, k=k, num_illegitimate_samples=num_illegitimate_samples, noise_scale=noise_scale
                        )
                        if X_train is None or X_test is None:
                            continue
                        
                        for nu in nu_values:
                            for gamma in gamma_values:
                                for kernel in kernel_values:
                                    model_name = f"svdd_{csv_file}_{k}_{num_illegitimate_samples}_{noise_scale}_{nu}_{gamma}_{kernel}"
                                    svdd_model = train_svdd(X_train, y_train, nu, gamma_values, kernel_values)
                                    if svdd_model:
                                        precision, recall, f1, far, frr, cm, eer = evaluate_model(svdd_model, X_test, y_test)
                                        metrics_results_svdd.append({
                                            'file': csv_file,
                                            'k': k,
                                            'num_illegitimate_samples': num_illegitimate_samples,
                                            'noise_scale': noise_scale,
                                            'nu': nu,
                                            'gamma': gamma,
                                            'kernel': kernel,
                                            'precision': precision,
                                            'recall': recall,
                                            'f1': f1,
                                            'far': far,
                                            'frr': frr,
                                            'cm': cm,
                                            'eer': eer
                                        })
                                        save_model(svdd_model, model_name)
                                        print(f"Evaluated SVDD model: {model_name}")
                                    else:
                                        print(f"Failed to train SVDD model: {model_name}")

                        ocknn_model = train_ocknn(X_train)
                        if ocknn_model:
                            precision, recall, f1, far, frr, cm, eer = evaluate_model(ocknn_model, X_test, y_test)
                            metrics_results_ocknn.append({
                                'file': csv_file,
                                'k': k,
                                'num_illegitimate_samples': num_illegitimate_samples,
                                'noise_scale': noise_scale,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'far': far,
                                'frr': frr,
                                'cm': cm,
                                'eer': eer
                            })
                            save_model(ocknn_model, f"ocknn_{csv_file}_{k}_{num_illegitimate_samples}_{noise_scale}")
                            print(f"Evaluated OCKNN model: {csv_file}_{k}_{num_illegitimate_samples}_{noise_scale}")
                        else:
                            print(f"Failed to train OCKNN model: {csv_file}_{k}_{num_illegitimate_samples}_{noise_scale}")

    # Save metrics results
    svdd_metrics_df = pd.DataFrame(metrics_results_svdd)
    ocknn_metrics_df = pd.DataFrame(metrics_results_ocknn)
    svdd_metrics_df.to_csv(os.path.join(plots_base_directory, 'svdd_metrics_results.csv'), index=False)
    ocknn_metrics_df.to_csv(os.path.join(plots_base_directory, 'ocknn_metrics_results.csv'), index=False)

    # Plot distributions
    plot_distributions(
        {
            'far': [result['far'] for result in metrics_results_svdd],
            'frr': [result['frr'] for result in metrics_results_svdd],
            'eer': [result['eer'] for result in metrics_results_svdd]
        },
        'SVDD'
    )
    
    plot_distributions(
        {
            'far': [result['far'] for result in metrics_results_ocknn],
            'frr': [result['frr'] for result in metrics_results_ocknn],
            'eer': [result['eer'] for result in metrics_results_ocknn]
        },
        'OCKNN'
    )

if __name__ == "__main__":
    main()

