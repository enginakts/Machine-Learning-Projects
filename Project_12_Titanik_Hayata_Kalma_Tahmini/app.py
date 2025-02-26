from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# modeli yükle
with open('titanic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():

    #form verilerini al
    pclass = int(request.form['pclass'])
    sex = int(request.form['sex'])
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])
    embarked = int(request.form['embarked'])

    # Tahmin içiin vrileri hazırla
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

    # Tahmin yap
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    # Sonucu hazırla
    survival = "Hayatta Kalır" if prediction[0] == 1 else "Hayatta Kalamaz"
    probability = round(probability[0][1] * 100, 2)

    return render_template('result.html', 
                         prediction=survival, 
                         probability=probability)

if __name__ == '__main__':
    app.run(debug=True)