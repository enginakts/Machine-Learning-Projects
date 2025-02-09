from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('gold_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        spx = float(request.form['spx'])
        uso = float(request.form['uso'])
        slv = float(request.form['slv'])
        eur_usd = float(request.form['eur_usd'])

        # Make prediction
        features = np.array([[spx, uso, slv, eur_usd]])
        prediction = model.predict(features)[0]
        
        # Calculate accuracy (you can use the R2 score from your model)
        accuracy = 0.98  # Replace with your model's actual accuracy

        return render_template('result.html', 
                             prediction=round(prediction, 2),
                             accuracy=round(accuracy * 100, 2))
    
    except Exception as e:
        return render_template('index.html', error="Invalid input. Please check your values.")

if __name__ == '__main__':
    app.run(debug=True) 