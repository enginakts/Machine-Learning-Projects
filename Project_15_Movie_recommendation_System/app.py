from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Veri setini yükle
df = pd.read_csv('movies.csv')

# TF-IDF Vectorizer'ı hazırla
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['keywords'].fillna(''))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['keywords']
        
        # Kullanıcı girdisini vektörize et
        user_vector = tfidf.transform([user_input])
        
        # Benzerlik skorlarını hesapla
        sim_scores = cosine_similarity(user_vector, tfidf_matrix)
        
        
        # En benzer 5 filmi bul
        movie_indices = sim_scores[0].argsort()[-5:][::-1]
        recommended_movies = df.iloc[movie_indices]
        
        # Benzerlik yüzdelerini hesapla
        similarity_percentages = sim_scores[0][movie_indices] * 100
        
        return render_template('result.html', 
                             movies=zip(recommended_movies.iterrows(), similarity_percentages))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 