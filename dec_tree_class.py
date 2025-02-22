from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target column
df['species'] = iris.target

# Show the first 5 rows
print(df.head())

from sklearn.model_selection import train_test_split

# Features (X) and target (y)
X = df.drop(columns=['species'])  # Input features
y = df['species']  # Output labels

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split completed.")

from sklearn.tree import DecisionTreeClassifier

# Create the Decision Tree model
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

print("Model training completed.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the first 10 predictions
print("Predictions:", y_pred[:10])

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {accuracy:.2f}")

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3)  # Limit depth to prevent overfitting
