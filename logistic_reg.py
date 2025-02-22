from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
data = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # Add target column

# Show first few rows
print(df.head())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop(columns=['target'])  # All columns except 'target'
y = df['target']  # Target column

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model training complete!")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the predictions
print("Predicted Labels:", y_pred)

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
