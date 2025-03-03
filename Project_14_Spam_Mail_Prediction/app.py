from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Flask uygulamasını başlat
app = Flask(__name__)

# Kaydedilmiş modeli yükle
with open('spam_mail_model.pkl', 'rb') as file:
    model = pickle.load(file)

# TF-IDF vektörleştiriciyi yükle
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    spam_messages = []
    if request.method == 'POST':
        emails = request.form['emails'].split('\n')  # Kullanıcının girdiği her satır bir e-posta olarak alınır
        email_features = vectorizer.transform(emails)  # TF-IDF dönüşümü
        predictions = model.predict(email_features)  # Model ile tahmin yap
        
        # Spam olanları listele
        spam_messages = [emails[i] for i in range(len(emails)) if predictions[i] == 0]
    
    return render_template('index.html', spam_messages=spam_messages)

if __name__ == '__main__':
    app.run(debug=True)
