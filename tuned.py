import zipfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import loguniform
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Extract dataset from zip file
def extract_zip(zip_path, extract_dir):
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted zip contents to '{extract_dir}'")

# 2. Load dataset
def load_cmapss_data(path):
    column_names = ['unit_number', 'time_in_cycles'] + \
                   [f'op_setting_{i}' for i in range(1, 4)] + \
                   [f'sensor_measurement_{i}' for i in range(1, 22)]
    try:
        df = pd.read_csv(path, sep='\s+', header=None)
    except:
        df = pd.read_csv(path, sep=' ', header=None)
    df.columns = column_names
    return df

# 3. Prepare features and labels for RUL prediction
def prepare_data(df):
    rul = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    rul.columns = ['unit_number', 'max_cycles']
    df = df.merge(rul, on='unit_number')
    df['RUL'] = df['max_cycles'] - df['time_in_cycles']
    df.drop(['max_cycles'], axis=1, inplace=True)
    features = df.drop(columns=['unit_number', 'time_in_cycles', 'RUL'])
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_df['RUL'] = df['RUL'].values
    X = scaled_df.drop(columns=['RUL'])
    y = scaled_df['RUL']
    return train_test_split(X, y, test_size=0.2, random_state=42), scaler

# 4. Train traditional ML models with timing
def train_models(X_train, y_train, X_val, y_val):
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        results.append([name, rmse, mae, r2, train_time])

        print(f"--- {name} ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Training Time: {train_time:.2f} seconds\n")

        plt.figure(figsize=(6, 4))
        plt.scatter(y_val[:100], y_pred[:100], alpha=0.6, color='teal')
        plt.plot([0, max(y_val[:100])], [0, max(y_val[:100])], color='red', linestyle='--')
        plt.xlabel('Actual RUL')
        plt.ylabel('Predicted RUL')
        plt.title(f'{name}: Actual vs Predicted RUL')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'MAE', 'R² Score', 'Training Time'])
    return models, results_df

# 5. Feature importance from Random Forest
def plot_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# 6. Hyperparameter tuning with timing
def hyperparameter_tuning(X_train, y_train):
    tuned_models = {}
    tuning_results = []

    # RandomForest tuning
    rf = RandomForestRegressor(random_state=42)
    rf_param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    start_time = time.time()
    rf_search = RandomizedSearchCV(rf, rf_param_dist, n_iter=10, cv=3, 
                                 scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    tune_time = time.time() - start_time
    tuned_models['RandomForest'] = rf_search.best_estimator_
    print(f"RandomForest best params: {rf_search.best_params_}")
    print(f"Tuning Time: {tune_time:.2f} seconds\n")
    tuning_results.append(['RandomForest', tune_time])

    # Ridge tuning
    ridge = Ridge()
    ridge_param_dist = {'alpha': loguniform(1e-3, 10)}
    start_time = time.time()
    ridge_search = RandomizedSearchCV(ridge, ridge_param_dist, n_iter=10, cv=3, 
                                    scoring='neg_mean_squared_error', random_state=42)
    ridge_search.fit(X_train, y_train)
    tune_time = time.time() - start_time
    tuned_models['Ridge'] = ridge_search.best_estimator_
    print(f"Ridge best alpha: {ridge_search.best_params_}")
    print(f"Tuning Time: {tune_time:.2f} seconds\n")
    tuning_results.append(['Ridge', tune_time])

    # Lasso tuning
    lasso = Lasso(max_iter=10000)
    lasso_param_dist = {'alpha': loguniform(1e-4, 1)}
    start_time = time.time()
    lasso_search = RandomizedSearchCV(lasso, lasso_param_dist, n_iter=10, cv=3, 
                                    scoring='neg_mean_squared_error', random_state=42)
    lasso_search.fit(X_train, y_train)
    tune_time = time.time() - start_time
    tuned_models['Lasso'] = lasso_search.best_estimator_
    print(f"Lasso best alpha: {lasso_search.best_params_}")
    print(f"Tuning Time: {tune_time:.2f} seconds\n")
    tuning_results.append(['Lasso', tune_time])

    return tuned_models, pd.DataFrame(tuning_results, columns=['Model', 'Tuning Time'])

# 7. Prepare data for LSTM
def prepare_lstm_data(df, seq_length=30):
    feature_cols = [col for col in df.columns if col.startswith('op_setting_') or col.startswith('sensor_measurement_')]
    data = df[feature_cols + ['RUL']].copy()

    X, y = [], []
    unit_ids = df['unit_number'].unique()

    for unit in unit_ids:
        unit_data = data[df['unit_number'] == unit]
        values = unit_data[feature_cols].values
        labels = unit_data['RUL'].values

        for i in range(len(unit_data) - seq_length):
            X.append(values[i:i+seq_length])
            y.append(labels[i+seq_length])

    return np.array(X), np.array(y)

# 8. Build and train LSTM model with timing
def build_and_train_lstm(X_train, y_train, X_val, y_val, epochs=30, batch_size=64):
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    start_time = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=1)
    train_time = time.time() - start_time
    
    return model, history, train_time

# 9. Evaluate LSTM with timing
def evaluate_lstm(model, X_val, y_val, train_time):
    start_time = time.time()
    y_pred = model.predict(X_val).flatten()
    predict_time = time.time() - start_time
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print("--- LSTM ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Prediction Time: {predict_time:.4f} seconds per sample\n")

    plt.figure(figsize=(6, 4))
    plt.scatter(y_val[:100], y_pred[:100], alpha=0.6, color='purple')
    plt.plot([0, max(y_val[:100])], [0, max(y_val[:100])], color='red', linestyle='--')
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.title('LSTM: Actual vs Predicted RUL')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return rmse, mae, r2, train_time

# --- Main execution ---
if __name__ == "__main__":
    # Data preparation
    zip_path = 'archive (2).zip'
    extract_dir = 'cmapss_data'
    extract_zip(zip_path, extract_dir)

    # Find train file path
    train_file_path = None
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file == 'train_FD001.txt':
                train_file_path = os.path.join(root, file)
                break

    train_df = load_cmapss_data(train_file_path)
    (X_train, X_val, y_train, y_val), scaler = prepare_data(train_df)

    # Train baseline models
    print("Training baseline models...\n")
    models, baseline_results = train_models(X_train, y_train, X_val, y_val)
    print("\nBaseline Model Results:")
    print(baseline_results.to_markdown(index=False))

    # Feature Importance
    print("\nPlotting feature importance from RandomForest...")
    plot_feature_importance(models['RandomForest'], X_train.columns)

    # Hyperparameter tuning
    print("\nStarting hyperparameter tuning...")
    tuned_models, tuning_times = hyperparameter_tuning(X_train, y_train)
    print("\nHyperparameter Tuning Times:")
    print(tuning_times.to_markdown(index=False))

    # Evaluate tuned models
    print("\nEvaluating tuned models on validation set...")
    tuned_results = []
    for name, model in tuned_models.items():
        start_time = time.time()
        y_pred = model.predict(X_val)
        predict_time = time.time() - start_time
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        tuned_results.append([name, rmse, mae, r2, predict_time/len(X_val)])

        print(f"{name} Tuned Model:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Prediction Time: {predict_time/len(X_val):.6f} seconds per sample\n")

        plt.figure(figsize=(6, 4))
        plt.scatter(y_val[:100], y_pred[:100], alpha=0.6, color='orange')
        plt.plot([0, max(y_val[:100])], [0, max(y_val[:100])], color='red', linestyle='--')
        plt.xlabel('Actual RUL')
        plt.ylabel('Predicted RUL')
        plt.title(f'{name} Tuned: Actual vs Predicted RUL')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    tuned_results_df = pd.DataFrame(tuned_results, 
                                   columns=['Model', 'RMSE', 'MAE', 'R² Score', 'Prediction Time'])
    print("Tuned Model Results:")
    print(tuned_results_df.to_markdown(index=False))

    # LSTM Model
    print("\nPreparing data for LSTM model...")
    seq_length = 30
    X_lstm, y_lstm = prepare_lstm_data(train_df, seq_length=seq_length)
    X_lstm_train, X_lstm_val, y_lstm_train, y_lstm_val = train_test_split(
        X_lstm, y_lstm, test_size=0.2, random_state=42)

    print("\nTraining LSTM model...")
    lstm_model, history, lstm_train_time = build_and_train_lstm(
        X_lstm_train, y_lstm_train, X_lstm_val, y_lstm_val, epochs=30, batch_size=64)

    # Plot training history
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Training History')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate LSTM
    lstm_rmse, lstm_mae, lstm_r2, lstm_time = evaluate_lstm(
        lstm_model, X_lstm_val, y_lstm_val, lstm_train_time)

    # Final comparison
    final_results = baseline_results.copy()
    final_results = final_results.append({
        'Model': 'LSTM',
        'RMSE': lstm_rmse,
        'MAE': lstm_mae,
        'R² Score': lstm_r2,
        'Training Time': lstm_time
    }, ignore_index=True)

    print("\nFinal Model Comparison:")
    print(final_results.to_markdown(index=False))
