import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
import time

# Load dataset
df = pd.read_csv('dataset_folder/CMaps/train_FD001.txt', sep='\s+', header=None)
df.columns = ['unit_number', 'time_in_cycles', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + \
             [f'sensor_measurement_{i}' for i in range(1, 22)]

# Compute RUL
rul = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
rul.columns = ['unit_number', 'max_cycles']
df = df.merge(rul, on='unit_number')
df['RUL'] = df['max_cycles'] - df['time_in_cycles']
df.drop(['max_cycles'], axis=1, inplace=True)

# Normalize features
features = df.columns.difference(['unit_number', 'time_in_cycles', 'RUL'])
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Sequence generation
SEQ_LEN = 20

def gen_sequence(id_df, seq_length, features):
    data_array = id_df[features].values
    seqs = []
    for start, stop in zip(range(0, len(data_array) - seq_length), range(seq_length, len(data_array))):
        seqs.append(data_array[start:stop])
    return np.array(seqs)

def gen_labels(id_df, seq_length):
    label_array = id_df['RUL'].values
    return label_array[seq_length:]

X, y, groups = [], [], []

for engine_id in df['unit_number'].unique():
    engine_df = df[df['unit_number'] == engine_id]
    seqs = gen_sequence(engine_df, SEQ_LEN, features)
    labels = gen_labels(engine_df, SEQ_LEN)
    X.extend(seqs)
    y.extend(labels)
    groups.extend([engine_id] * len(seqs))

X, y, groups = np.array(X), np.array(y), np.array(groups)

# Split data
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Build Bi-LSTM model
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.3),
    BatchNormalization(),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
start_time = time.time()
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[lr_scheduler], verbose=1)
training_time = time.time() - start_time

# Evaluate
y_pred = model.predict(X_test).flatten()
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
mae = np.mean(np.abs(y_test - y_pred))
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print(f"Improved Bi-LSTM Results:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR² Score: {r2:.4f}\nTraining Time: {training_time:.2f} seconds")


# --- Step 0: Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --- Step 1: Load and Prepare CMAPSS Dataset ---
df = pd.read_csv('dataset_folder/CMaps/train_FD001.txt', sep='\s+', header=None)
df.dropna(axis=1, inplace=True)
df.columns = ['unit', 'time', 'op1', 'op2', 'op3'] + [f'sensor{i}' for i in range(1, df.shape[1] - 5 + 1)]

# Calculate RUL
rul_df = df.groupby('unit')['time'].max().reset_index()
rul_df.columns = ['unit', 'max_time']
df = df.merge(rul_df, on='unit')
df['RUL'] = df['max_time'] - df['time']
df.drop(columns=['max_time'], inplace=True)

# Drop low-variance or irrelevant sensors (based on prior domain knowledge)
drop_cols = ['op1', 'op2', 'op3'] + [f'sensor{i}' for i in [1, 5, 6, 10, 16, 18, 19]]
df.drop(columns=drop_cols, inplace=True)

# Normalize features (exclude unit, time, RUL)
feature_cols = df.columns.difference(['unit', 'time', 'RUL'])
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# --- Step 2: Create Sequences for Time Series ---
def create_sequences(df, seq_len=30):
    X, y = [], []
    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit]
        for i in range(len(unit_df) - seq_len):
            X.append(unit_df[feature_cols].iloc[i:i+seq_len].values)
            y.append(unit_df['RUL'].iloc[i + seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(df, seq_len=30)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Define Models ---
def build_uni_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_bi_lstm(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Step 4: Train and Evaluate Models ---
def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name="Model"):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=50, batch_size=64,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks, verbose=1)
    training_time = time.time() - start_time

    y_pred = model.predict(X_val).flatten()
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"\n🔍 {model_name} Results:")
    print(f"  RMSE      : {rmse:.4f}")
    print(f"  MAE       : {mae:.4f}")
    print(f"  R² Score  : {r2:.4f}")
    print(f"  Time Taken: {training_time:.2f} seconds")

    return y_pred, history

# Build models
uni_model = build_uni_lstm(X_train.shape[1:])
bi_model = build_bi_lstm(X_train.shape[1:])

# Train and evaluate Uni-LSTM
y_pred_uni, history_uni = train_and_evaluate(uni_model, X_train, y_train, X_val, y_val, model_name="Uni-directional LSTM")

# Train and evaluate Bi-LSTM
y_pred_bi, history_bi = train_and_evaluate(bi_model, X_train, y_train, X_val, y_val, model_name="Bidirectional LSTM")

# --- Step 5: Plot Results Side-by-Side ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred_uni, alpha=0.6, label='Predicted')
plt.plot([0, max(y_val)], [0, max(y_val)], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Uni-directional LSTM: Actual vs Predicted RUL')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_val, y_pred_bi, alpha=0.6, label='Predicted')
plt.plot([0, max(y_val)], [0, max(y_val)], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Bidirectional LSTM: Actual vs Predicted RUL')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
