# week3_ml_notebook.py (can be converted to .ipynb)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# Load cleaned data
df = pd.read_csv('cleaned_ev_dataset.csv')

# Feature selection
X = df[['Hour','ChargingDuration']]
y = df['EnergyConsumption']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf_params = {'n_estimators':[50,100], 'max_depth':[5,10,None]}
rf_cv = GridSearchCV(rf, rf_params, cv=3, scoring='neg_mean_absolute_error')
rf_cv.fit(X_train, y_train)
rf_best = rf_cv.best_estimator_
y_pred_rf = rf_best.predict(X_test)

# Evaluation function
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

lr_metrics = evaluate(y_test, y_pred_lr)
rf_metrics = evaluate(y_test, y_pred_rf)

print('Linear Regression Metrics (MAE, RMSE, R2):', lr_metrics)
print('Random Forest Metrics (MAE, RMSE, R2):', rf_metrics)

# Choose best model
best_model = rf_best if rf_metrics[0] < lr_metrics[0] else lr

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/ev_energy_model.pkl')
print('Saved best model to models/ev_energy_model.pkl')

# Save predictions & actuals for plotting
results = pd.DataFrame({'y_true': y_test, 'y_pred_lr': y_pred_lr, 'y_pred_rf': y_pred_rf})
results.to_csv('models/predictions.csv', index=False)

# Plots
os.makedirs('plots', exist_ok=True)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf, alpha=0.6)
plt.xlabel('Actual Energy (kWh)')
plt.ylabel('Predicted Energy (kWh)')
plt.title('Actual vs Predicted (Random Forest)')
plt.savefig('plots/actual_vs_predicted.png')
plt.close()

residuals = y_test - y_pred_rf
plt.figure(figsize=(8,6))
plt.hist(residuals, bins=30)
plt.title('Residuals Distribution (Random Forest)')
plt.xlabel('Residual (kWh)')
plt.savefig('plots/residuals.png')
plt.close()

print('Plots saved to plots/ folder')
