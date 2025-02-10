from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Modeli yükle
model = pickle.load(open('heart_disease_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Form verilerini al
    features = [float(x) for x in request.form.values()]
    features = [np.array(features)]
    
    # Tahmin yap
    prediction = model.predict(features)
    
    # Sonucu hazırla
    output = "Kalp hastalığı riski var" if prediction[0] == 1 else "Kalp hastalığı riski yok"
    
    return render_template('result.html', prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True) 