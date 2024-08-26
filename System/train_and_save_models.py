import sys
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
import logging

logging.basicConfig(filename='train_and_save_models_log.txt',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
        
        #Debugging prints
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

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

    # Ensure gamma_values and kernel_values are lists
    if not isinstance(gamma_values, list):
        gamma_values = [gamma_values]
    if not isinstance(kernel_values, list):
        kernel_values = [kernel_values]

    param_grid = {
        'kernel': kernel_values,
        'gamma': gamma_values,
        'nu': [nu]
    }

    svdd = OneClassSVM()
    grid_search = GridSearchCV(svdd, param_grid, cv=3, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

def enroll_user(user_id, user_data_path, best_svdd_params, other_users_directory=None):
    """
    Enroll a user by training and saving their SVDD and OCKNN models.
    """
    logging.info(f"Enrolling user: {user_id}")
    logging.info(f"User data path: {user_data_path}")
    logging.info(f"SVDD parameters: {best_svdd_params}")
    logging.info(f"Other users directory: {other_users_directory}")
    try:
        # Load and preprocess data for the new user
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            user_data_path, k=10, num_illegitimate_samples=0, noise_scale=0.05
        )

        if X_train is None or X_test is None:
            raise ValueError("Failed to preprocess data")

        # Generate illegitimate samples based on other users' data if provided
        if other_users_directory:
            # Collect all other users' data
            all_other_users_features = []
            for csv_file in os.listdir(other_users_directory):
                if csv_file.endswith('.csv'):
                    csv_file_path = os.path.join(other_users_directory, csv_file)
                    other_features_df = pd.read_csv(csv_file_path, header=None)
                    other_features_df = other_features_df[other_features_df.iloc[:, 1] != 'E']
                    other_features_df.iloc[:, 2] = pd.to_numeric(other_features_df.iloc[:, 2], errors='coerce')
                    other_features_df.iloc[:, 4] = pd.to_numeric(other_features_df.iloc[:, 4], errors='coerce')
                    other_features_df.iloc[:, 5] = pd.to_numeric(other_features_df.iloc[:, 5], errors='coerce')
                    other_features_df = other_features_df.dropna()
                    other_features_df = extract_features(other_features_df)
                    all_other_users_features.append(other_features_df.values)
            
            if all_other_users_features:
                all_other_users_features = np.vstack(all_other_users_features)
                num_illegitimate_samples = min(40, all_other_users_features.shape[0])
                illegitimate_features = generate_illegitimate_samples(
                    all_other_users_features, num_samples=num_illegitimate_samples, noise_scale=0.05
                )
                illegitimate_labels = np.ones(num_illegitimate_samples) * -1
                X_train = np.vstack((X_train, illegitimate_features))
                y_train = np.hstack((y_train, illegitimate_labels))

        # Convert single values to lists if needed
        gamma_values = best_svdd_params.get('gamma_values', [0.1])
        kernel_values = best_svdd_params.get('kernel_values', ['rbf'])

        # Train SVDD model using best parameters
        svdd_model = train_svdd(X_train, y_train, 
                                nu=best_svdd_params['nu'], 
                                gamma_values=gamma_values, 
                                kernel_values=kernel_values)
        
        # Train OCKNN model
        ocknn_model = train_ocknn(X_train)

        # Save the trained models
        if svdd_model:
            svdd_model_path = os.path.join(models_base_directory, f'{user_id}_svdd_model.pkl')
            save_model(svdd_model, f'{user_id}_svdd_model')

        if ocknn_model:
            ocknn_model_path = os.path.join(models_base_directory, f'{user_id}_ocknn_model.pkl')
            save_model(ocknn_model, f'{user_id}_ocknn_model')

        # Evaluate models and plot metrics
        if X_test is not None and y_test is not None:
            # Evaluate SVDD model
            svdd_precision, svdd_recall, svdd_f1, svdd_far, svdd_frr, svdd_cm, svdd_eer = evaluate_model(svdd_model, X_test, y_test)
            # Evaluate OCKNN model
            ocknn_precision, ocknn_recall, ocknn_f1, ocknn_far, ocknn_frr, ocknn_cm, ocknn_eer = evaluate_model(ocknn_model, X_test, y_test)
            
             # Print metrics
            logging.info(f"SVDD Metrics for {user_id}:")
            logging.info(f"Precision: {svdd_precision}")
            logging.info(f"Recall: {svdd_recall}")
            logging.info(f"F1 Score: {svdd_f1}")
            logging.info(f"FAR: {svdd_far}")
            logging.info(f"FRR: {svdd_frr}")
            logging.info(f"EER: {svdd_eer}")
            sys.stdout.flush() 
            
            logging.info(f"OCKNN Metrics for {user_id}:")
            logging.info(f"Precision: {ocknn_precision}")
            logging.info(f"Recall: {ocknn_recall}")
            logging.info(f"F1 Score: {ocknn_f1}")
            logging.info(f"FAR: {ocknn_far}")
            logging.info(f"FRR: {ocknn_frr}")
            logging.info(f"EER: {ocknn_eer}")
            sys.stdout.flush() 

             # Plot and save confusion matrices
            logging.info(f"SVDD Confusion Matrix for {user_id}:")
            logging.info(svdd_cm)
            plot_confusion_matrix(svdd_cm, 'SVDD')
            
            logging.info(f"OCKNN Confusion Matrix for {user_id}:")
            logging.info(ocknn_cm)
            plot_confusion_matrix(ocknn_cm, 'OCKNN')
            
            # Plot and save distributions
            metrics = {
                'far': [svdd_far, ocknn_far],
                'frr': [svdd_frr, ocknn_frr],
                'eer': [svdd_eer, ocknn_eer]
            }
            plot_distributions(metrics, 'Model')

        return svdd_model_path, ocknn_model_path

    except Exception as e:
        logging.error(f"Error during user enrollment for {user_id}: {e}")
        return None, None



def authenticate_user(user_id, user_data_path):
    """
    Authenticate a user by loading the saved models and making predictions.
    """
    try:
        # Load models
        svdd_model_path = os.path.join(models_base_directory, f'{user_id}_svdd_model.pkl')
        ocknn_model_path = os.path.join(models_base_directory, f'{user_id}_ocknn_model.pkl')
        
        svdd_model = joblib.load(svdd_model_path)
        ocknn_model = joblib.load(ocknn_model_path)

        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            user_data_path, k=10, num_illegitimate_samples=40, noise_scale=0.05
        )

        if X_train is None or X_test is None:
            raise ValueError("Failed to preprocess data")

        # Predict using SVDD model
        svdd_predictions = svdd_model.predict(X_test)
        # Predict using OCKNN model
        ocknn_predictions = ocknn_model.predict(X_test)

        # Evaluation
        svdd_precision, svdd_recall, svdd_f1, svdd_far, svdd_frr, svdd_cm, svdd_eer = evaluate_model(svdd_model, X_test, y_test)
        ocknn_precision, ocknn_recall, ocknn_f1, ocknn_far, ocknn_frr, ocknn_cm, ocknn_eer = evaluate_model(ocknn_model, X_test, y_test)
        
        # Return results
        return {
            'svdd': {
                'precision': svdd_precision,
                'recall': svdd_recall,
                'f1': svdd_f1,
                'far': svdd_far,
                'frr': svdd_frr,
                'cm': svdd_cm,
                'eer': svdd_eer
            },
            'ocknn': {
                'precision': ocknn_precision,
                'recall': ocknn_recall,
                'f1': ocknn_f1,
                'far': ocknn_far,
                'frr': ocknn_frr,
                'cm': ocknn_cm,
                'eer': ocknn_eer
            }
        }

    except Exception as e:
        logging.error(f"Error during user authentication for {user_id}: {e}")
        return None

def train_user_model(user_id, user_data_path, best_svdd_params):
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        user_data_path, k=10, num_illegitimate_samples=40, noise_scale=0.05
    )

    # Log the shapes of the datasets
    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    if X_train is None or X_test is None:
        return None, None

    # Train SVDD model using best parameters
    svdd_model = train_svdd(X_train, y_train, 
                            nu=best_svdd_params['nu'], 
                            gamma_values=best_svdd_params['gamma_values'], 
                            kernel_values=best_svdd_params['kernel_values'])
    
    # Train OCKNN model
    ocknn_model = train_ocknn(X_train)

    # Save the trained models
    svdd_model_path = None
    ocknn_model_path = None
    
    if svdd_model:
        svdd_model_path = os.path.join(models_base_directory, f'{user_id}_svdd_model.pkl')
        save_model(svdd_model, f'{user_id}_svdd_model')
    
    if ocknn_model:
        ocknn_model_path = os.path.join(models_base_directory, f'{user_id}_ocknn_model.pkl')
        save_model(ocknn_model, f'{user_id}_ocknn_model')

    return svdd_model_path, ocknn_model_path



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

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legitimate', 'Illegitimate'],
                yticklabels=['Legitimate', 'Illegitimate'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(plots_base_directory, f'{model_name}_confusion_matrix.png'))
    plt.close()

def main():
    # Hyperparameter tuning example
    best_svdd_model = None
    best_svdd_metrics = None
    best_svdd_params = {}
    
    k_values = [5, 10, 15]  # Example values, adjust as needed
    num_illegitimate_samples_values = [20, 40, 60]  # Example values, adjust as needed
    noise_scales = [0.01, 0.05, 0.1]  # Example values, adjust as needed
    nu_values = [0.1, 0.5, 0.9]  # Example values, adjust as needed
    gamma_values = [0.1, 1, 10]  # Example values, adjust as needed
    kernel_values = ['linear', 'rbf']  # Example values, adjust as needed

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
                                    svdd_model = train_svdd(X_train, y_train, nu, gamma, kernel)
                                    if svdd_model:
                                        precision, recall, f1, far, frr, cm, eer = evaluate_model(svdd_model, X_test, y_test)
                                        metrics = {
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
                                        }
                                        # Update best model based on metrics
                                        if best_svdd_metrics is None or f1 > best_svdd_metrics['f1']:
                                            best_svdd_model = svdd_model
                                            best_svdd_metrics = metrics
                                            best_svdd_params = {
                                                'nu': nu,
                                                'gamma_values': gamma_values,
                                                'kernel_values': kernel_values
                                            }
    
    if best_svdd_model:
        print(f"Best SVDD Model found with F1 score: {best_svdd_metrics['f1']}")
        # Save or use the best SVDD model as needed
        save_model(best_svdd_model, 'best_svdd_model')

if __name__ == "__main__":
    main()
