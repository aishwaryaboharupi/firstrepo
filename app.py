from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import joblib

# Load the XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgboost_regression.json")

# Load the Flight Path Prediction model
flight_model = joblib.load("flight_path_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Machine Learning Model API is Running!"

@app.route('/predict_xgb', methods=['POST'])
def predict_xgb():
    """Predict using the XGBoost model"""
    try:
        data = request.get_json()  # Get JSON data from request

        # Convert input JSON to DataFrame
        df = pd.DataFrame([data])

        # Ensure the input matches the expected feature order for XGBoost
        feature_order = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
        df = df[feature_order]

        # Make a prediction
        prediction = xgb_model.predict(df)

        # Return prediction as JSON response
        return jsonify({"prediction": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_flight', methods=['POST'])
def predict_flight():
    """Predict using the Flight Path model"""
    try:
        data = request.get_json()  # Get JSON data from request

        # Convert input JSON to DataFrame
        df = pd.DataFrame([data])

        # Ensure correct feature order for flight model
        flight_features = ["Velocity (m/s)", "Altitude (km)", "Drag Coefficient", "Angle of Attack (°)"]
        df = df[flight_features]

        # Make a prediction
        prediction = flight_model.predict(df)

        # Return predictions as JSON
        return jsonify({
            "Next Velocity (m/s)": float(prediction[0][0]),
            "Next Altitude (km)": float(prediction[0][1])
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
