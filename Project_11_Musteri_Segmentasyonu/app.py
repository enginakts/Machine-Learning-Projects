from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('customer_segmentation_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Get values from the form
        income = float(request.form['income'])
        spending = float(request.form['spending'])
        
        # Make prediction
        features = np.array([[income, spending]])
        prediction = int(model.predict(features)[0]) + 1
        
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True) 