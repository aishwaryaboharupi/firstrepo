import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset (Boston Housing dataset)
df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")

# Features and target
X = df.drop(columns=["medv"])  # Drop target column (house price)
y = df["medv"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Save Model
model.save_model("xgboost_regression.json")

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],  
    'max_depth': [3, 5, 7],  
    'learning_rate': [0.01, 0.1, 0.2]  
}

grid_search = GridSearchCV(xgb.XGBRegressor(), param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
