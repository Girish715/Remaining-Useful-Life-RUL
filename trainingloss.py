# Install required packages with error handling
try:
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
    from tensorflow.keras.layers import (GRU, LSTM, Dense, Dropout, BatchNormalization)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
except ImportError as e:
    !pip install numpy pandas matplotlib seaborn scikit-learn tensorflow --quiet
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
    from tensorflow.keras.layers import (GRU, LSTM, Dense, Dropout, BatchNormalization)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("tab10")

# =============================================
# 1. Data Preparation
# =============================================
print("Preparing data...")

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

# Create sequences
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(data)

# Create separate scalers
X_scaler = StandardScaler()
y_scaler = StandardScaler()

# Scale features and target
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Train/val/test split (60/20/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Reshape for LSTM/GRU
X_train_seq = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_seq = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_seq = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# =============================================
# 2. GRU Model Implementation
# =============================================
print("\nBuilding GRU Model...")

def build_gru_model(input_shape):
    model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        GRU(32),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='mse',
                 metrics=['mae'])
    return model

gru_model = build_gru_model((X_train_seq.shape[1], X_train_seq.shape[2]))

# Train with early stopping
history = gru_model.fit(X_train_seq, y_train,
                       epochs=200,
                       batch_size=16,
                       validation_data=(X_val_seq, y_val),
                       callbacks=[
                           EarlyStopping(patience=15, restore_best_weights=True),
                           ReduceLROnPlateau(factor=0.5, patience=5)
                       ],
                       verbose=1)

# =============================================
# 3. Evaluation and Visualization
# =============================================
def evaluate_model(model, X_test, y_test, model_name):
    # Predict
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform
    y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.6)
    plt.plot([min(y_test_orig), max(y_test_orig)], 
             [min(y_test_orig), max(y_test_orig)], 'r--')
    plt.title(f'{model_name} - Actual vs Predicted RUL', fontsize=14)
    plt.xlabel('Actual RUL', fontsize=12)
    plt.ylabel('Predicted RUL', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # Plot residuals
    residuals = y_test_orig - y_pred_orig
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_orig, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'{model_name} - Residual Plot', fontsize=14)
    plt.xlabel('Predicted RUL', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # Training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training History - Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training History - MAE', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('MAE', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return rmse, mae, r2

# Evaluate on test set
print("\nEvaluating GRU Model on Test Set...")
rmse, mae, r2 = evaluate_model(gru_model, X_test_seq, y_test, "GRU Model")

print("\nTest Set Metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Save model and scalers
print("\nSaving model and scalers...")
gru_model.save('gru_rul_model.h5')
joblib.dump(X_scaler, 'x_scaler.pkl')
joblib.dump(y_scaler, 'y_scaler.pkl')
print("Model and scalers saved successfully!")
