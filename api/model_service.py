import joblib
import pickle
import pandas as pd
import os

class ModelService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.loaded = False
    
    def load(self):
        self.model = joblib.load('../models/model.pkl')
        self.scaler = joblib.load('../models/scaler.pkl')
        with open('../models/feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        self.loaded = True
    
    def predict(self, data):
        if not self.loaded:
            self.load()
        
        df = pd.DataFrame([data])[self.feature_names]
        X = self.scaler.transform(df)
        
        prob = float(self.model.predict_proba(X)[0, 1])
        pred = int(self.model.predict(X)[0])
        
        return {'probability': prob, 'prediction': pred}

service = ModelService()