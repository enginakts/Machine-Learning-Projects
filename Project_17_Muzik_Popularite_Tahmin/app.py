from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = {
            'danceability': float(request.form['danceability']),
            'energy': float(request.form['energy']),
            'valence': float(request.form['valence']),
            'acousticness': float(request.form['acousticness']),
            'tempo': float(request.form['tempo']),
            'speechiness': float(request.form['speechiness']),
            'liveness': float(request.form['liveness'])
        }
        
        # Create input array for model
        input_array = np.array([[
            features['danceability'],
            features['energy'],
            features['valence'],
            features['acousticness'],
            features['tempo'],
            features['speechiness'],
            features['liveness']
        ]])
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'message': f'Predicted Popularity Score: {int(prediction)}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 