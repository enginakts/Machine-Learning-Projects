from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('insurance_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    r2_score = 0.75  # Bu değeri gerçek R2 skorunuzla değiştirin
    
    if request.method == 'POST':
        # Get values from the form
        age = float(request.form['age'])
        sex = 1 if request.form['sex'] == 'female' else 0
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = 1 if request.form['smoker'] == 'yes' else 0
        region = {
            'southeast': 0,
            'southwest': 1,
            'northeast': 2,
            'northwest': 3
        }[request.form['region']]
        
        # Make prediction
        features = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(features)[0]
        prediction = round(prediction, 2)
    
    return render_template('index.html', prediction=prediction, r2_score=r2_score)

if __name__ == '__main__':
    app.run(debug=True) 