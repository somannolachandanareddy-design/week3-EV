import pandas as pd
import matplotlib.pyplot as plt
import os

# Load predictions
pred = pd.read_csv('models/predictions.csv')

os.makedirs('plots', exist_ok=True)

# Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(pred['y_true'], pred['y_pred'], alpha=0.6)
plt.xlabel('Actual Energy (kWh)')
plt.ylabel('Predicted Energy (kWh)')
plt.title('Actual vs Predicted')
plt.savefig('plots/actual_vs_predicted.png')
plt.close()

# Residuals
residuals = pred['y_true'] - pred['y_pred']
plt.figure(figsize=(8,6))
plt.hist(residuals, bins=30)
plt.title('Residuals Distribution')
plt.xlabel('Residual (kWh)')
plt.savefig('plots/residuals.png')
plt.close()

print('Saved plots to plots/ folder')
