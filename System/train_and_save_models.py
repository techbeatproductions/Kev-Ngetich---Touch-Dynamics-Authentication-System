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
    """
    Extract features from the input DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: Feature DataFrame
    """
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

def generate_illegitimate_samples(X, num_samples, noise_scale):
    num_samples = min(num_samples, X.shape[0])
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=X.shape)
    illegitimate_samples = X + noise
    indices = np.random.choice(X.shape[0], num_samples, replace=False)
    return illegitimate_samples[indices]

def load_and_preprocess_data(csv_file_path, k=10, num_illegitimate_samples=40, noise_scale=0.05):
    """
    Load and preprocess data from a CSV file.

    Parameters:
    csv_file_path (str): Path to the CSV file
    k (int): Number of features to select
    num_illegitimate_samples (int): Number of illegitimate samples to generate
    noise_scale (float): Noise scale for generating illegitimate samples

    Returns:
    X_train, X_test, y_train, y_test (np.ndarray): Preprocessed data
    """
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
    """
    Train an SVDD model with hyperparameter tuning.

    Parameters:
    X_train (np.ndarray): Training data
    y_train (np.ndarray): Training labels
    nu (float): Nu value for SVDD
    gamma_values (list): List of gamma values to try
    kernel_values (list): List of kernel values to try

    Returns:
    OneClassSVM: Trained SVDD model
    """
    if X_train is None or y_train is None:
        return None

    param_grid = {
        'kernel': kernel_values,
        'gamma': gamma_values,
        'nu': [nu]
    }
    
    svdd = OneClassSVM()
    grid_search = GridSearchCV(svdd, param_grid, cv=3, scoring='f1')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

    """
    Train an SVDD model with hyperparameter tuning.

    Parameters:
    X_train (np.ndarray): Training data
    y_train (np.ndarray): Training labels
    nu (float): Nu value for SVDD
    gamma_values (list): List of gamma values to try
    kernel_values (list): List of kernel values to try

    Returns:
    OneClassSVM: Trained SVDD model
    """
    if X_train is None or y_train is None:
        return None

    param_grid = {
        'kernel': kernel_values,
        'gamma': gamma_values,
        'nu': [nu]
    }
    
    svdd = OneClassSVM()
    grid_search = GridSearchCV(svdd, param_grid, cv=3, scoring='f1')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_ocknn(X_train):
    """
    Train an OCKNN model.

    Parameters:
    X_train (np.ndarray): Training data

    Returns:
    LocalOutlierFactor: Trained OCKNN model
    """
    ocknn = LocalOutlierFactor(novelty=True)
    ocknn.fit(X_train)
    return ocknn

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model on the test data.

    Parameters:
    model (OneClassSVM or LocalOutlierFactor): Trained model
    X_test (np.ndarray): Test data
    y_test (np.ndarray): Test labels

    Returns:
    precision, recall, f1, far, frr, cm, eer (float): Evaluation metrics
    """
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
    """
    Plot the distributions of FAR, FRR, and EER.

    Parameters:
    metrics (dict): Evaluation metrics
    model_name (str): Model name
    """
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

def plot_confusion_matrix(cm, model_name):
    """
    Plot the confusion matrix.

    Parameters:
    cm (np.ndarray): Confusion matrix
    model_name (str): Model name
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Legitimate', 'Illegitimate'], 
                yticklabels=['Legitimate', 'Illegitimate'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(plots_base_directory, f'{model_name}_confusion_matrix.png'))
    plt.close()

def save_model(model, filename):
    """
    Save the trained model to a file.

    Parameters:
    model (object): Trained model
    filename (str): Filename for saving the model

    Returns:
    str: Path to the saved model
    """
    model_path = os.path.join(models_base_directory, filename)
    joblib.dump(model, model_path)
    return model_path

def train_user_model(user_id, user_data_path, best_svdd_params):
    """
    Train SVDD and OCKNN models for a specific user.

    Parameters:
    user_id (str): User ID
    user_data_path (str): Path to the user data CSV
    best_svdd_params (dict): Best parameters for SVDD model
    """
    X_train, X_test, y_train, y_test = load_and_preprocess_data(user_data_path)

    if X_train is not None and y_train is not None:
        # Train SVDD model with best parameters
        svdd_model = train_svdd(X_train, y_train, **best_svdd_params)
        precision, recall, f1, far, frr, cm, eer = evaluate_model(svdd_model, X_test, y_test)
        
        print(f"SVDD Model Performance for User {user_id}:")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"FAR: {far}")
        print(f"FRR: {frr}")
        print("Confusion Matrix:")
        print(cm)
        
        plot_confusion_matrix(cm, f'SVDD_User_{user_id}')
        plot_distributions({'far': [far], 'frr': [frr], 'eer': [eer]}, f'SVDD_User_{user_id}')
        svdd_model_path = save_model(svdd_model, f'svdd_model_{user_id}.pkl')

        # Train OCKNN model
        ocknn_model = train_ocknn(X_train)
        ocknn_precision, ocknn_recall, ocknn_f1, ocknn_far, ocknn_frr, ocknn_cm, ocknn_eer = evaluate_model(ocknn_model, X_test, y_test)

        print(f"OCKNN Model Performance for User {user_id}:")
        print(f"Precision: {ocknn_precision}")
        print(f"Recall: {ocknn_recall}")
        print(f"F1 Score: {ocknn_f1}")
        print(f"FAR: {ocknn_far}")
        print(f"FRR: {ocknn_frr}")
        print("Confusion Matrix:")
        print(ocknn_cm)
        
        plot_confusion_matrix(ocknn_cm, f'OCKNN_User_{user_id}')
        plot_distributions({'far': [ocknn_far], 'frr': [ocknn_frr], 'eer': [ocknn_eer]}, f'OCKNN_User_{user_id}')
        ocknn_model_path = save_model(ocknn_model, f'ocknn_model_{user_id}.pkl')

        return svdd_model_path, ocknn_model_path
    else:
        return None, None

    """
    Train SVDD and OCKNN models for a specific user.

    Parameters:
    user_id (str): User ID
    user_data (str): Path to the user data CSV
    best_svdd_params (dict): Best parameters for SVDD model
    """
    X_train, X_test, y_train, y_test = load_and_preprocess_data(user_data)

    if X_train is not None and y_train is not None:
        # Train SVDD model with best parameters
        svdd_model = train_svdd(X_train, y_train, **best_svdd_params)
        precision, recall, f1, far, frr, cm, eer = evaluate_model(svdd_model, X_test, y_test)
        
        print(f"SVDD Model Performance for User {user_id}:")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"FAR: {far}")
        print(f"FRR: {frr}")
        print("Confusion Matrix:")
        print(cm)
        
        plot_confusion_matrix(cm, f'SVDD_User_{user_id}')
        plot_distributions({'far': [far], 'frr': [frr], 'eer': [eer]}, f'SVDD_User_{user_id}')
        save_model(svdd_model, f'svdd_model_{user_id}.pkl')

        # Train OCKNN model
        ocknn_model = train_ocknn(X_train)
        ocknn_precision, ocknn_recall, ocknn_f1, ocknn_far, ocknn_frr, ocknn_cm, ocknn_eer = evaluate_model(ocknn_model, X_test, y_test)

        print(f"OCKNN Model Performance for User {user_id}:")
        print(f"Precision: {ocknn_precision}")
        print(f"Recall: {ocknn_recall}")
        print(f"F1 Score: {ocknn_f1}")
        print(f"FAR: {ocknn_far}")
        print(f"FRR: {ocknn_frr}")
        print("Confusion Matrix:")
        print(ocknn_cm)
        
        plot_confusion_matrix(ocknn_cm, f'OCKNN_User_{user_id}')
        plot_distributions({'far': [ocknn_far], 'frr': [ocknn_frr], 'eer': [ocknn_eer]}, f'OCKNN_User_{user_id}')
        save_model(ocknn_model, f'ocknn_model_{user_id}.pkl')

def main():
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    # Define hyperparameter ranges
    nu_values = [0.1, 0.5, 0.9]
    gamma_values = [0.001, 0.01, 0.1, 1, 10]
    kernel_values = ['linear', 'rbf']
    
    best_svdd_params = None
    best_svdd_metrics = {
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'far': 0,
        'frr': 0,
        'eer': float('inf')
    }
    
    for csv_file in tqdm(csv_files, desc="Processing Files"):
        csv_file_path = os.path.join(directory, csv_file)
        
        for nu in nu_values:
            for gamma in gamma_values:
                for kernel in kernel_values:
                    print(f"Processing {csv_file} with SVDD parameters: nu={nu}, gamma={gamma}, kernel={kernel}")
                    X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_file_path)
                    
                    if X_train is not None and y_train is not None:
                        svdd_model = train_svdd(X_train, y_train, nu, [gamma], [kernel])
                        
                        if svdd_model:
                            precision, recall, f1, far, frr, cm, eer = evaluate_model(svdd_model, X_test, y_test)
                            
                            if eer < best_svdd_metrics['eer']:
                                best_svdd_metrics = {
                                    'precision': precision,
                                    'recall': recall,
                                    'f1': f1,
                                    'far': far,
                                    'frr': frr,
                                    'eer': eer
                                }
                                best_svdd_params = {
                                    'nu': nu,
                                    'gamma': gamma,
                                    'kernel': kernel
                                }
                                
    print(f"Best SVDD Parameters: {best_svdd_params}")
    print(f"Best SVDD Metrics: {best_svdd_metrics}")
    
    # Save the best SVDD model
    for csv_file in csv_files:
        user_id = os.path.splitext(csv_file)[0]
        user_data_path = os.path.join(directory, csv_file)
        train_user_model(user_id, user_data_path, best_svdd_params)

if __name__ == "__main__":
    main()
