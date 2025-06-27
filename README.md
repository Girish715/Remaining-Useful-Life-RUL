# Remaining Useful Life (RUL) Prediction

This project focuses on predicting the Remaining Useful Life (RUL) of industrial components using both classical machine learning and deep learning techniques. The goal is to enable predictive maintenance through accurate estimation of how long equipment can operate before failure.

---

## Project Overview

- **Objective**: Predict how much useful life remains in equipment based on sensor readings.
- **Data**: Multivariate time-series sensor data from industrial machines.
- **Models Implemented**:
  - Linear Regression, Ridge, Lasso
  - Random Forest, XGBoost, LightGBM
  - LSTM, GRU, Bi-LSTM
  - Autoencoder + Regressor
  - Transformer-based Time Series Model

---

##  Key Features

- End-to-end pipeline: data cleaning, feature engineering, model training, evaluation
- Sequence generation for deep learning models
- Attention mechanisms in Transformer models
- Visualization of RUL predictions over time
- Evaluation metrics: MAE, RMSE, R² Score

---

## Tools & Libraries

- Python
- Scikit-learn
- TensorFlow / Keras
- XGBoost, LightGBM
- Matplotlib, Seaborn
- Pandas, NumPy

---

## Results

- Achieved consistent improvement moving from linear models to GRU and Transformer architectures.
- Transformer models excelled in capturing long-term dependencies and improving prediction stability.

---


yaml
Copy
Edit

---

## Acknowledgments

Special thanks to **Wu Jiyan** for mentorship and guidance throughout the internship.

---

## Author

Girish Kanna
