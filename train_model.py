import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load cleaned dataset
print('Loading cleaned_ev_dataset.csv...')
df = pd.read_csv('cleaned_ev_dataset.csv')

# Features and target
X = df[['Hour','ChargingDuration']]
y = df['EnergyConsumption']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}')

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/ev_energy_model.pkl')
print('Model saved to models/ev_energy_model.pkl')

# Save predictions
pred_df = X_test.copy()
pred_df['y_true'] = y_test
pred_df['y_pred'] = y_pred
pred_df.to_csv('models/predictions.csv', index=False)
print('Predictions saved to models/predictions.csv')
