import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (GRU, LSTM, Dense, Dropout, BatchNormalization, 
                                   Conv1D, MaxPooling1D, Flatten, Input, RepeatVector, 
                                   TimeDistributed, MultiHeadAttention, LayerNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed for reproducibility
np.random.seed(42)

# =============================================
# 1. Data Preparation
# =============================================
print("Preparing data...")

# Load data
data = np.loadtxt('RUL_FD001.txt')

# Create sequences for time series prediction
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # Number of previous time steps to use for prediction
X, y = create_sequences(data, seq_length)

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, 1)).reshape(-1, seq_length)
y = scaler.transform(y.reshape(-1, 1)).reshape(-1)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for models [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# =============================================
# 2. GRU Model
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
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Build and train GRU model
gru_model = build_gru_model((X_train.shape[1], X_train.shape[2]))
gru_history = gru_model.fit(X_train, y_train, 
                          epochs=100, 
                          batch_size=16, 
                          validation_data=(X_test, y_test),
                          callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                          verbose=0)

# Evaluate
gru_loss = gru_model.evaluate(X_test, y_test, verbose=0)
print(f"GRU Model Test Loss: {gru_loss:.4f}")

# =============================================
# 3. CNN-LSTM Hybrid Model
# =============================================
print("\nBuilding CNN-LSTM Model...")

def build_cnn_lstm_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Build and train CNN-LSTM model
cnn_lstm_model = build_cnn_lstm_model((X_train.shape[1], X_train.shape[2]))
cnn_lstm_history = cnn_lstm_model.fit(X_train, y_train, 
                                     epochs=100, 
                                     batch_size=16, 
                                     validation_data=(X_test, y_test),
                                     callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                                     verbose=0)

# Evaluate
cnn_lstm_loss = cnn_lstm_model.evaluate(X_test, y_test, verbose=0)
print(f"CNN-LSTM Model Test Loss: {cnn_lstm_loss:.4f}")

# =============================================
# 4. Autoencoder + Regressor Model
# =============================================
print("\nBuilding Autoencoder + Regressor Model...")

def build_autoencoder_regressor(input_shape, encoding_dim=8):
    # Encoder
    inputs = Input(shape=input_shape)
    encoded = LSTM(32, return_sequences=True)(inputs)
    encoded = LSTM(16)(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder (for autoencoder)
    decoded = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(16, return_sequences=True)(decoded)
    decoded = LSTM(32, return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(input_shape[1]))(decoded)
    
    # Autoencoder model
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Regressor model
    regressor_output = Dense(16, activation='relu')(encoded)
    regressor_output = Dense(1)(regressor_output)
    regressor = Model(inputs, regressor_output)
    regressor.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return autoencoder, regressor

# Build models
autoencoder, regressor = build_autoencoder_regressor((X_train.shape[1], X_train.shape[2]))

# First train the autoencoder
autoencoder.fit(X_train, X_train,
               epochs=50,
               batch_size=16,
               validation_data=(X_test, X_test),
               callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
               verbose=0)

# Then train the regressor
regressor_history = regressor.fit(X_train, y_train,
                                epochs=100,
                                batch_size=16,
                                validation_data=(X_test, y_test),
                                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                                verbose=0)

# Evaluate
autoencoder_loss = regressor.evaluate(X_test, y_test, verbose=0)
print(f"Autoencoder Regressor Model Test Loss: {autoencoder_loss:.4f}")

# =============================================
# 5. Transformer Model
# =============================================
print("\nBuilding Transformer Model...")

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    
    # Feed Forward Part
    y = Dense(ff_dim, activation="relu")(x)
    y = Dense(inputs.shape[-1])(y)
    y = Dropout(dropout)(y)
    y = LayerNormalization(epsilon=1e-6)(x + y)
    
    return y

def build_transformer_model(input_shape, head_size=32, num_heads=4, ff_dim=64, dropout=0.2):
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Transformer blocks
    for _ in range(2):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    # Final layers
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Build and train Transformer model
transformer_model = build_transformer_model((X_train.shape[1], X_train.shape[2]))
transformer_history = transformer_model.fit(X_train, y_train,
                                         epochs=100,
                                         batch_size=16,
                                         validation_data=(X_test, y_test),
                                         callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                                         verbose=0)

# Evaluate
transformer_loss = transformer_model.evaluate(X_test, y_test, verbose=0)
print(f"Transformer Model Test Loss: {transformer_loss:.4f}")

# =============================================
# 6. Visualization and Comparison
# =============================================
print("\nGenerating visualizations...")

# Plot training histories
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(gru_history.history['loss'], label='Train')
plt.plot(gru_history.history['val_loss'], label='Validation')
plt.title('GRU Model Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(cnn_lstm_history.history['loss'], label='Train')
plt.plot(cnn_lstm_history.history['val_loss'], label='Validation')
plt.title('CNN-LSTM Model Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(regressor_history.history['loss'], label='Train')
plt.plot(regressor_history.history['val_loss'], label='Validation')
plt.title('Autoencoder Regressor Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(transformer_history.history['loss'], label='Train')
plt.plot(transformer_history.history['val_loss'], label='Validation')
plt.title('Transformer Model Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# Plot predictions comparison
plt.figure(figsize=(15, 10))

models = {
    'GRU': gru_model,
    'CNN-LSTM': cnn_lstm_model,
    'Autoencoder+Regressor': regressor,
    'Transformer': transformer_model
}

for i, (name, model) in enumerate(models.items(), 1):
    plt.subplot(2, 2, i)
    y_pred = model.predict(X_test)
    plt.plot(y_test, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.title(f'{name} Predictions')
    plt.legend()

plt.tight_layout()
plt.show()

# Compare model performances
plt.figure(figsize=(10, 5))
model_names = list(models.keys())
losses = [gru_loss, cnn_lstm_loss, autoencoder_loss, transformer_loss]
plt.bar(model_names, losses)
plt.title('Model Comparison (Test Loss)')
plt.ylabel('MSE Loss')
plt.show()

# Print final comparison
print("\nFinal Model Comparison:")
for name, loss in zip(model_names, losses):
    print(f"{name}: {loss:.4f} MSE")
