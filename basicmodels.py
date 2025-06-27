
import zipfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

zip_path = 'archive (2).zip'
extract_dir = 'cmapss_data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

train_file_path = None
for root, dirs, files in os.walk(extract_dir):
    for file in files:
        if file == 'train_FD001.txt':
            train_file_path = os.path.join(root, file)
            break
def load_cmapss_data(path):
    column_names = ['unit_number', 'time_in_cycles'] + \
                   [f'op_setting_{i}' for i in range(1, 4)] + \
                   [f'sensor_measurement_{i}' for i in range(1, 22)]
    df = pd.read_csv(path, sep='\s+', header=None)
    df.columns = column_names
    return df

train_df = load_cmapss_data(train_file_path)
rul = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
rul.columns = ['unit_number', 'max_cycles']
train_df = train_df.merge(rul, on='unit_number')
train_df['RUL'] = train_df['max_cycles'] - train_df['time_in_cycles']
train_df.drop(['max_cycles'], axis=1, inplace=True)
features = train_df.drop(columns=['unit_number', 'time_in_cycles', 'RUL'])
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_df['RUL'] = train_df['RUL']
X = scaled_df.drop(columns=['RUL'])
y = scaled_df['RUL']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(n_estimators=100)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    results.append([name, rmse, mae, r2])

    print(f"--- {name} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}\n")
    plt.figure(figsize=(6, 4))
    plt.scatter(y_val[:100], y_pred[:100], alpha=0.6, color='teal')
    plt.plot([0, max(y_val[:100])], [0, max(y_val[:100])], color='red', linestyle='--')
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.title(f'{name}: Actual vs Predicted RUL')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'MAE', 'R² Score'])
print(results_df)
