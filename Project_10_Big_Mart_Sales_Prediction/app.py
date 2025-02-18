from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
model = pickle.load(open('big_mart_sales_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    features = [float(x) for x in request.form.values()]
    features = np.array([features])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Calculate accuracy (using R2 score from your training)
    accuracy = 0.956  # You can replace this with your model's R2 score
    
    return render_template('result.html', 
                         prediction=round(prediction[0], 2),
                         accuracy=accuracy * 100)

if __name__ == '__main__':
    app.run(debug=True) 