from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('credit_card_fraud_model.pkl')

@app.route('/', methods=['POST'])
def predict():
    try:
        # Extract data from the request
        data = request.json
        # Convert input data to the format required by the model
        features = np.array(data['features']).reshape(1, -1)
         
        # Make prediction
        prediction = model.predict(features)
        
        # Return the prediction as a JSON response
        
        return str(prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000)
