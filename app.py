from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd

# Load the saved XGBoost model
model = xgb.XGBRegressor()
model.load_model("xgboost_regression.json")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "XGBoost Model API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get JSON data from request

        # Convert input JSON to DataFrame
        df = pd.DataFrame([data])

        # Ensure the input matches the expected feature order
        feature_order = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
        df = df[feature_order]

        # Make a prediction
        prediction = model.predict(df)

        # Return prediction as JSON response
        return jsonify({"prediction": float(prediction[0])})  # Convert NumPy float to Python float
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
