from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

try:
    # Load the saved model
    model = joblib.load('credit_card_fraud_model.pkl')
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Get values from the form
            features = []
            for i in range(1, 31):  # 30 features (Time, V1-V28, Amount)
                value = float(request.form.get(f'feature{i}', 0.0))
                features.append(value)
            
            # Convert to numpy array and reshape
            features = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features)[0]
            prediction = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
        except Exception as e:
            prediction = f"Hata oluştu: {e}"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    print("Flask uygulaması başlatılıyor...")
    app.run(debug=True, port=5000) 