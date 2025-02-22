import joblib
import numpy as np

# Load the saved model
model = joblib.load("salary_model.pkl")

# Predict salary for a given experience
experience = np.array([[5]])  # Change this value to test different inputs
predicted_salary = model.predict(experience)

print(f"Predicted Salary for {experience[0][0]} years of experience: {predicted_salary[0]}")
