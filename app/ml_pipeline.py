import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from datetime import datetime

class MLPipeline:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        
    def create_training_data(self):
        # Minimal training data for demonstration
        data = [
            ("My bill is too high", "billing"),
            ("I can't log into my account", "technical"),
            ("The app keeps crashing", "technical"),
            ("When is my payment due", "billing"),
            ("I'm very disappointed with service", "complaint"),
            ("Hello, I need help", "general"),
            ("Refund my money", "billing"),
            ("System error occurred", "technical")
        ]
        return pd.DataFrame(data, columns=['text', 'intent'])
    
    def train_model(self):
        with mlflow.start_run():
            # Create training data
            df = self.create_training_data()
            
            # Create pipeline
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000)),
                ('classifier', MultinomialNB())
            ])
            
            # Train model
            self.model.fit(df['text'], df['intent'])
            
            # Log model with MLflow
            mlflow.sklearn.log_model(self.model, "intent_classifier")
            mlflow.log_param("model_type", "MultinomialNB")
            mlflow.log_param("vectorizer", "TfidfVectorizer")
            mlflow.log_param("training_samples", len(df))
            
            # Calculate accuracy on training data (for demo)
            accuracy = self.model.score(df['text'], df['intent'])
            mlflow.log_metric("accuracy", accuracy)
            
            print(f"Model trained with accuracy: {accuracy}")
            return self.model
    
    def predict_intent(self, text):
        if self.model is None:
            self.train_model()
        
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def load_model(self, model_uri):
        self.model = mlflow.sklearn.load_model(model_uri)
        return self.model

if __name__ == "__main__":
    pipeline = MLPipeline()
    pipeline.train_model()