# Install required packages
!pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras-tuner lightgbm --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (GRU, LSTM, Dense, Dropout, BatchNormalization, 
                                   Conv1D, MaxPooling1D, Flatten, Input, RepeatVector, 
                                   TimeDistributed, MultiHeadAttention, LayerNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from keras_tuner import RandomSearch
import lightgbm as lgb

# Set random seed for reproducibility
np.random.seed(42)
plt.style.use('seaborn-v0_8')  # Updated seaborn style reference

# =============================================
# 1. Enhanced Data Preparation with Holdout Set
# =============================================
print("Preparing data with proper train/val/test split...")

# Generate synthetic data if file not found
try:
    data = np.loadtxt('RUL_FD001.txt')
except FileNotFoundError:
    print("Using synthetic data as RUL_FD001.txt not found")
    np.random.seed(42)
    trend = np.linspace(100, 0, 200)
    noise = np.random.normal(0, 5, 200)
    data = trend + noise
    data = np.clip(data, 0, None)  # Ensure no negative RUL

# Create sequences with enhanced features
def create_sequences(data, seq_length=15):
    X, y = [], []
    for i in range(len(data)-seq_length):
        window = data[i:i+seq_length]
        
        # Basic features
        features = [
            np.mean(window), np.std(window), np.min(window), np.max(window),
            np.percentile(window, 25), np.percentile(window, 50), np.percentile(window, 75),
            (window[-1] - window[0])/seq_length  # slope
        ]
        X.append(np.concatenate([window, features]))
        y.append(data[i+seq_length])
    
    return np.array(X), np.array(y)

X, y = create_sequences(data)

# Create proper train/val/test split (60/20/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Create separate scalers
X_scaler = StandardScaler()
y_scaler = StandardScaler()

# Scale features
X_train_scaled = X_scaler.fit_transform(X_train)
X_val_scaled = X_scaler.transform(X_val)
X_test_scaled = X_scaler.transform(X_test)

# Scale target
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Reshape for sequence models
def reshape_for_sequence(X):
    return X.reshape((X.shape[0], X.shape[1], 1))

X_train_seq = reshape_for_sequence(X_train_scaled)
X_val_seq = reshape_for_sequence(X_val_scaled)
X_test_seq = reshape_for_sequence(X_test_scaled)

# Save scalers for inference
joblib.dump(X_scaler, 'X_scaler.pkl')
joblib.dump(y_scaler, 'y_scaler.pkl')

# =============================================
# 2. Enhanced GRU Model with Visualization
# =============================================
print("\nBuilding and training GRU Model...")

def build_gru_model():
    model = Sequential([
        GRU(128, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    return model

gru_model = build_gru_model()
gru_history = gru_model.fit(
    X_train_seq, 
    y_train_scaled,
    epochs=200,
    batch_size=32,
    validation_data=(X_val_seq, y_val_scaled),
    callbacks=[
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ],
    verbose=1
)

# =============================================
# 3. Enhanced Transformer Model
# =============================================
print("\nBuilding Transformer Model...")

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Attention layer
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size, dropout=dropout
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    x = LayerNormalization(epsilon=1e-6)(attention_output + inputs)
    
    # Feed Forward layer
    y = Dense(ff_dim, activation="relu")(x)
    y = Dense(inputs.shape[-1])(y)
    y = Dropout(dropout)(y)
    y = LayerNormalization(epsilon=1e-6)(x + y)
    
    return y

def build_transformer_model(input_shape, head_size=64, num_heads=4, ff_dim=128, num_layers=2, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = inputs
    
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
    return model

transformer_model = build_transformer_model((X_train_seq.shape[1], X_train_seq.shape[2]))
transformer_history = transformer_model.fit(
    X_train_seq, 
    y_train_scaled,
    epochs=200,
    batch_size=32,
    validation_data=(X_val_seq, y_val_scaled),
    callbacks=[
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ],
    verbose=1
)

# =============================================
# 4. LightGBM Model for Comparison
# =============================================
# 4. LightGBM Model for Comparison (Corrected)
# =============================================
print("\nTraining LightGBM model for comparison...")

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=7,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20  # Moved from fit() to here
)

start_time = time.time()
lgb_model.fit(
    X_train_scaled, 
    y_train_scaled,
    eval_set=[(X_val_scaled, y_val_scaled)],
    eval_metric='l1',  # MAE
    verbose=100
)
lgb_time = time.time() - start_time

# =============================================
# 5. Comprehensive Evaluation and Visualization
# =============================================
def evaluate_and_visualize(model, X, y_true_scaled, model_name, is_sequence=True):
    # Predict
    if is_sequence:
        y_pred_scaled = model.predict(X)
    else:
        y_pred_scaled = model.predict(X)
    
    # Inverse transform
    y_true = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title(f'{model_name} - Actual vs Predicted RUL')
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.grid(True)
    plt.show()
    
    # Plot residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'{model_name} - Residual Plot')
    plt.xlabel('Predicted RUL')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()
    
    # Error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f'{model_name} - Error Distribution')
    plt.xlabel('Prediction Error')
    plt.grid(True)
    plt.show()
    
    return rmse, mae, r2, y_true, y_pred

# Evaluate all models on test set
print("\n=== Test Set Evaluation ===")
models = [
    ('GRU', gru_model, True),
    ('Transformer', transformer_model, True),
    ('LightGBM', lgb_model, False)
]

results = []
for name, model, is_seq in models:
    rmse, mae, r2, y_true, y_pred = evaluate_and_visualize(
        model, 
        X_test_seq if is_seq else X_test_scaled, 
        y_test_scaled, 
        name, 
        is_seq
    )
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    })

# Show results dataframe
results_df = pd.DataFrame(results)
print("\nFinal Model Comparison on Test Set:")
print(results_df.to_string(index=False))

# Plot training histories for deep learning models
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training History (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - Training History (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_training_history(gru_history, 'GRU Model')
plot_training_history(transformer_history, 'Transformer Model')

# Save best model
print("\nSaving best performing model...")
best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
if best_model_name == 'GRU':
    gru_model.save('best_gru_model.h5')
elif best_model_name == 'Transformer':
    transformer_model.save('best_transformer_model.h5')
else:
    joblib.dump(lgb_model, 'best_lgb_model.pkl')
print(f"Saved {best_model_name} as the best performing model.")

# =============================================
# 6. Error Analysis
# =============================================
print("\nAnalyzing error cases...")

# Get predictions from best model
if best_model_name in ['GRU', 'Transformer']:
    y_pred = gru_model.predict(X_test_seq) if best_model_name == 'GRU' else transformer_model.predict(X_test_seq)
else:
    y_pred = lgb_model.predict(X_test_scaled)

y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
errors = np.abs(y_test - y_pred)

# Plot worst predictions
worst_indices = np.argsort(errors)[-5:]  # Top 5 worst predictions
plt.figure(figsize=(12, 6))
for idx in worst_indices:
    plt.plot(X_test[idx, :15], '.-', label=f'Sample {idx} (True RUL: {y_test[idx]:.1f}, Pred: {y_pred[idx]:.1f})')
plt.title('Sensor Readings for Worst Predictions')
plt.xlabel('Time Step')
plt.ylabel('Sensor Value')
plt.legend()
plt.grid(True)
plt.show()

# Error by RUL range
rul_bins = np.linspace(min(y_test), max(y_test), 5)
bin_errors = []
for i in range(len(rul_bins)-1):
    mask = (y_test >= rul_bins[i]) & (y_test < rul_bins[i+1])
    bin_errors.append(np.mean(errors[mask]))

plt.figure(figsize=(10, 5))
plt.bar(range(len(bin_errors)), bin_errors)
plt.xticks(range(len(bin_errors)), 
           [f"{rul_bins[i]:.0f}-{rul_bins[i+1]:.0f}" for i in range(len(bin_errors))])
plt.title('Average Prediction Error by RUL Range')
plt.xlabel('RUL Range')
plt.ylabel('Mean Absolute Error')
plt.grid(True)
plt.show()
