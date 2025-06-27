
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time

# Load and preprocess dataset (same as before)
df = pd.read_csv('dataset_folder/CMaps/train_FD001.txt', sep='\s+', header=None)
df.dropna(axis=1, inplace=True)
df.columns = ['unit', 'time', 'op1', 'op2', 'op3'] + [f'sensor{i}' for i in range(1, df.shape[1] - 5 + 1)]
rul_df = df.groupby('unit')['time'].max().reset_index()
rul_df.columns = ['unit', 'max_time']
df = df.merge(rul_df, on='unit')
df['RUL'] = df['max_time'] - df['time']
df.drop(columns=['max_time'], inplace=True)
drop_cols = ['op1', 'op2', 'op3'] + [f'sensor{i}' for i in [1, 5, 6, 10, 16, 18, 19]]
df.drop(columns=drop_cols, inplace=True)
feature_cols = df.columns.difference(['unit', 'time', 'RUL'])
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

X = df[feature_cols].values
y = df['RUL'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix for train and validation
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'eta': 0.1,
    'seed': 42
}

evals = [(dtrain, 'train'), (dval, 'eval')]

start_time = time.time()
model = xgb.train(params, dtrain, num_boost_round=100,
                  evals=evals,
                  early_stopping_rounds=10,
                  verbose_eval=True)
training_time = time.time() - start_time

# Predict
y_pred = model.predict(dval)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"\nXGBoost Regression Results:")
print(f"RMSE          : {rmse:.4f}")
print(f"MAE           : {mae:.4f}")
print(f"R² Score      : {r2:.4f}")
print(f"Training Time : {training_time:.2f} seconds")

# Plot
plt.figure(figsize=(7, 6))
plt.scatter(y_val, y_pred, alpha=0.6)
plt.plot([0, max(y_val)], [0, max(y_val)], color='red', linestyle='--')
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('XGBoost Regression: Actual vs Predicted RUL')
plt.grid(True)
plt.show()

