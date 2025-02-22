import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("libraries imported successfully")

df = pd.read_csv("salary_data.csv")

print(df.head())

# check dataset info
print(df.info())

# summary stats
print(df.describe())

# scatter plot of YearsExperience vs salary
plt.scatter(df["YearsExperience"], df["Salary"], color="blue")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()

#Linear Regression
from sklearn.model_selection import train_test_split

# define features (X) and target (y)
X = df[["YearsExperience"]]
y = df["Salary"]

#split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split successful!")

model = LinearRegression()
model.fit(X_train, y_train)

print("Model training complete!")

#make predictions on the test set
y_pred = model.predict(X_test)

#print predictions
print("Predictions:", y_pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

import joblib

# Save the model
joblib.dump(model, "salary_model.pkl")
print("Model saved as salary_model.pkl")
