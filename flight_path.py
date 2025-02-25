import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load trajectory dataset
df = pd.read_csv("hypersonic_flight_trajectory.csv")

# Rename columns to match expected feature names
column_renames = {"Angle of Attack (°)": "Angle of Attack (deg)"}
df.rename(columns=column_renames, inplace=True)

# Define features and target variables
FEATURES = ["Velocity (m/s)", "Altitude (km)", "Drag Coefficient", "Angle of Attack (deg)"]
TARGET_VARS = ["Next Velocity (m/s)", "Next Altitude (km)"]

# Ensure all required features exist
missing_features = [feature for feature in FEATURES if feature not in df.columns]
if missing_features:
    raise KeyError(f"Missing features in dataset: {missing_features}")

X = df[FEATURES]
y = df[TARGET_VARS]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save the trained model (for later use in simulations)
joblib.dump(model, "flight_path_model.pkl")

print("Flight path prediction model trained successfully!")
