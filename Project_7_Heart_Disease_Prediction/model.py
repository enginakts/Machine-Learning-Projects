import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

def train_model():
    # Veri setini yükle
    heart_data = pd.read_csv('heart_disease_data.csv')
    
    # Features ve target'ı ayır
    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']
    
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Model eğitimi
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    # Modeli kaydet
    pickle.dump(model, open('heart_disease_model.pkl', 'wb'))

if __name__ == "__main__":
    train_model() 