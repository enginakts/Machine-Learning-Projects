from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Model yükleme
model = joblib.load('calories_prediction_model.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Formdan gelen verileri al
            gender = int(request.form['gender'])  # 0: Erkek, 1: Kadın
            age = float(request.form['age'])
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            duration = float(request.form['duration'])
            heart_rate = float(request.form['heart_rate'])
            body_temp = float(request.form['body_temp'])

            # Tahmin için verileri hazırla
            features = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])
            
            # Tahmin yap
            prediction = model.predict(features)[0]
            prediction = round(prediction, 2)

        except Exception as e:
            prediction = "Hata oluştu: Lütfen tüm alanları doğru formatta doldurun."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True) 