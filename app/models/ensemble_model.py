import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import mlflow
from typing import Dict, List, Tuple

class TransformerEnsemble(nn.Module):
    def __init__(self, model_names: List[str], num_classes: int = 4):
        super().__init__()
        self.models = nn.ModuleList()
        self.tokenizers = []
        
        for model_name in model_names:
            model = AutoModel.from_pretrained(model_name)
            self.models.append(model)
            self.tokenizers.append(AutoTokenizer.from_pretrained(model_name))
        
        self.classifier = nn.Linear(len(model_names) * 768, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        embeddings = []
        
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                pooled = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(pooled)
        
        combined = torch.cat(embeddings, dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return torch.softmax(logits, dim=1)

class AdvancedQueryClassifier:
    def __init__(self):
        self.intent_model = None
        self.sentiment_model = None
        self.priority_model = None
        self.ensemble_model = None
        self.feature_extractors = {}
        
    def build_ensemble(self):
        """Build ensemble of multiple models for robust predictions"""
        # Traditional ML models
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        lr_model = LogisticRegression(random_state=42)
        
        # Ensemble voting classifier
        self.ensemble_model = VotingClassifier([
            ('rf', rf_model),
            ('lr', lr_model)
        ], voting='soft')
        
        # Transformer ensemble
        transformer_models = [
            'distilbert-base-uncased',
            'roberta-base'
        ]
        self.transformer_ensemble = TransformerEnsemble(transformer_models)
        
    def extract_advanced_features(self, text: str) -> Dict:
        """Extract comprehensive features from text"""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'urgency_keywords': sum(1 for word in ['urgent', 'asap', 'immediately', 'emergency'] if word in text.lower()),
            'sentiment_keywords_pos': sum(1 for word in ['good', 'great', 'excellent', 'satisfied'] if word in text.lower()),
            'sentiment_keywords_neg': sum(1 for word in ['bad', 'terrible', 'awful', 'disappointed'] if word in text.lower())
        }
        return features
    
    def predict_priority(self, text: str, intent: str, sentiment: str) -> Tuple[str, float]:
        """Predict query priority based on multiple factors"""
        features = self.extract_advanced_features(text)
        
        priority_score = 0.0
        
        # Intent-based scoring
        intent_weights = {'complaint': 0.8, 'technical': 0.6, 'billing': 0.4, 'general': 0.2}
        priority_score += intent_weights.get(intent, 0.2)
        
        # Sentiment-based scoring
        if sentiment == 'NEGATIVE':
            priority_score += 0.3
        
        # Feature-based scoring
        priority_score += features['urgency_keywords'] * 0.2
        priority_score += features['exclamation_count'] * 0.1
        priority_score += features['caps_ratio'] * 0.15
        
        # Normalize to 0-1 range
        priority_score = min(priority_score, 1.0)
        
        if priority_score >= 0.7:
            return "HIGH", priority_score
        elif priority_score >= 0.4:
            return "MEDIUM", priority_score
        else:
            return "LOW", priority_score
    
    def predict_with_confidence(self, text: str) -> Dict:
        """Advanced prediction with confidence intervals and uncertainty quantification"""
        # Multiple predictions for uncertainty estimation
        predictions = []
        
        for _ in range(5):  # Monte Carlo sampling
            # Add slight noise for uncertainty estimation
            noisy_text = text  # In practice, you'd add dropout or other noise
            
            # Get predictions from ensemble
            intent_pred = self._predict_intent_ensemble(noisy_text)
            sentiment_pred = self._predict_sentiment_ensemble(noisy_text)
            
            predictions.append({
                'intent': intent_pred,
                'sentiment': sentiment_pred
            })
        
        # Calculate confidence and uncertainty
        intent_confidence = self._calculate_confidence([p['intent'] for p in predictions])
        sentiment_confidence = self._calculate_confidence([p['sentiment'] for p in predictions])
        
        # Final prediction (majority vote)
        final_intent = max(set([p['intent'][0] for p in predictions]), 
                          key=[p['intent'][0] for p in predictions].count)
        final_sentiment = max(set([p['sentiment'][0] for p in predictions]), 
                             key=[p['sentiment'][0] for p in predictions].count)
        
        # Priority prediction
        priority, priority_score = self.predict_priority(text, final_intent, final_sentiment)
        
        return {
            'intent': final_intent,
            'intent_confidence': intent_confidence,
            'sentiment': final_sentiment,
            'sentiment_confidence': sentiment_confidence,
            'priority': priority,
            'priority_score': priority_score,
            'uncertainty_score': 1 - min(intent_confidence, sentiment_confidence)
        }
    
    def _predict_intent_ensemble(self, text: str) -> Tuple[str, float]:
        """Predict intent using ensemble approach"""
        # Simplified for demo - would use actual trained models
        intents = ['billing', 'technical', 'complaint', 'general']
        scores = np.random.dirichlet([1, 1, 1, 1])  # Placeholder
        
        max_idx = np.argmax(scores)
        return intents[max_idx], scores[max_idx]
    
    def _predict_sentiment_ensemble(self, text: str) -> Tuple[str, float]:
        """Predict sentiment using ensemble approach"""
        sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        scores = np.random.dirichlet([1, 1, 1])  # Placeholder
        
        max_idx = np.argmax(scores)
        return sentiments[max_idx], scores[max_idx]
    
    def _calculate_confidence(self, predictions: List[Tuple[str, float]]) -> float:
        """Calculate prediction confidence based on consistency"""
        labels = [pred[0] for pred in predictions]
        most_common = max(set(labels), key=labels.count)
        confidence = labels.count(most_common) / len(labels)
        return confidence