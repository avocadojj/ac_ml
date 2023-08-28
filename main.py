from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Function for blended prediction
def blended_prediction(X):
    return (
        0.4 * xgb_model.predict(X) +
        0.3 * lgbm_model.predict(X) +
        0.3 * catboost_model.predict(X)
    )[0]

# Load models and preprocessing pipeline
try:
    xgb_model = joblib.load('xgb_model.pkl')
    lgbm_model = joblib.load('lgbm_model.pkl')
    catboost_model = joblib.load('catboost_model.pkl')
except Exception as e:
    print(f"Error loading model or preprocessor: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request as JSON
        data = request.get_json()
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([data])
    
        # Make blended prediction
        prediction = blended_prediction(df)

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
def home():
    return "Welcome to the Prediction API. Use the /predict endpoint to make predictions."

if __name__ == '__main__':
    app.run(debug=True)
