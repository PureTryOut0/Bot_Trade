from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

class SentimentAnalyzer:
    def __init__(self):
        # Load or initialize the sentiment model
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        
        # Initialize local model for backup
        self.local_model = None
        self.vectorizer = None
        self.load_local_model()
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'positive': 0.7,
            'neutral': 0.5,
            'negative': 0.7
        }

    def load_local_model(self):
        try:
            model_path = "models/local_sentiment_model.joblib"
            vectorizer_path = "models/vectorizer.joblib"
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                self.local_model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
        except Exception as e:
            print(f"Error loading local model: {e}")

    def analyze_sentiment(self, text, source="unknown"):
        try:
            # Try Hugging Face model first
            result = self.pipeline(text)[0]
            sentiment = result['label']
            confidence = result['score']
            
            # Convert to numeric score
            if sentiment == 'LABEL_2':  # Positive
                score = 1.0 * confidence
            elif sentiment == 'LABEL_1':  # Neutral
                score = 0.0
            else:  # Negative
                score = -1.0 * confidence
                
            return {
                'sentiment': sentiment,
                'score': score,
                'confidence': confidence,
                'source': source,
                'method': 'hf'
            }
            
        except Exception as e:
            print(f"HF model failed: {e}")
            # Fall back to local model if available
            if self.local_model and self.vectorizer:
                try:
                    features = self.vectorizer.transform([text])
                    pred = self.local_model.predict_proba(features)[0]
                    sentiment = np.argmax(pred)
                    confidence = pred[sentiment]
                    
                    if sentiment == 2:  # Positive
                        score = 1.0 * confidence
                    elif sentiment == 1:  # Neutral
                        score = 0.0
                    else:  # Negative
                        score = -1.0 * confidence
                        
                    return {
                        'sentiment': sentiment,
                        'score': score,
                        'confidence': confidence,
                        'source': source,
                        'method': 'local'
                    }
                except Exception as e:
                    print(f"Local model failed: {e}")
                    return None
            return None

    def update_model(self, new_data, labels):
        """Update the local model with new labeled data"""
        try:
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(max_features=10000)
                
            features = self.vectorizer.fit_transform(new_data)
            self.local_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
            self.local_model.fit(features, labels)
            
            # Save updated model
            joblib.dump(self.local_model, "models/local_sentiment_model.joblib")
            joblib.dump(self.vectorizer, "models/vectorizer.joblib")
            
            return True
        except Exception as e:
            print(f"Error updating model: {e}")
            return False

    def get_confidence_score(self, sentiment_result):
        """Calculate a combined confidence score based on multiple factors"""
        if not sentiment_result:
            return 0.0
            
        base_confidence = sentiment_result['confidence']
        source_weight = {
            'reddit': 0.9,
            'twitter': 0.85,
            'news': 0.95,
            'unknown': 0.7
        }[sentiment_result['source']]
        
        return base_confidence * source_weight

    def analyze_batch(self, texts, sources):
        """Analyze multiple texts with source information"""
        results = []
        for text, source in zip(texts, sources):
            result = self.analyze_sentiment(text, source)
            if result:
                result['confidence_score'] = self.get_confidence_score(result)
                results.append(result)
        return results
