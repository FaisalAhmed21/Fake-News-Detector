from flask import Flask, render_template, request, jsonify
import re
import csv
import os
import json
import math
from collections import Counter, defaultdict
import numpy as np

app = Flask(__name__)

# Enhanced text preprocessing function
def wordopt(text):
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', "", text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove newlines
    text = re.sub(r'\n', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

# Enhanced TF-IDF Vectorizer with better feature selection
class EnhancedTfidfVectorizer:
    def __init__(self, max_features=2000, min_df=2, max_df=0.95):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary_ = {}
        self.idf_ = {}
        self.fitted = False
    
    def fit(self, texts):
        # Count document frequency for each word
        doc_freq = defaultdict(int)
        word_counts = defaultdict(int)
        total_docs = len(texts)
        
        for text in texts:
            words = text.split()
            unique_words = set(words)
            for word in unique_words:
                doc_freq[word] += 1
            for word in words:
                word_counts[word] += 1
        
        # Filter words based on document frequency
        filtered_words = {}
        for word, count in word_counts.items():
            df = doc_freq[word]
            df_ratio = df / total_docs
            
            if df >= self.min_df and df_ratio <= self.max_df:
                filtered_words[word] = count
        
        # Select top features
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, count in sorted_words[:self.max_features]]
        
        # Create vocabulary
        self.vocabulary_ = {word: idx for idx, word in enumerate(top_words)}
        
        # Calculate enhanced IDF with smoothing
        for word in self.vocabulary_:
            df = doc_freq.get(word, 0)
            # Enhanced IDF calculation with smoothing
            self.idf_[word] = math.log((total_docs + 1) / (df + 1)) + 1
        
        self.fitted = True
        return self
    
    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)
    
    def transform(self, texts):
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        features = []
        for text in texts:
            words = text.split()
            word_freq = Counter(words)
            
            # Create feature vector with enhanced TF-IDF
            vector = [0] * len(self.vocabulary_)
            max_freq = max(word_freq.values()) if word_freq else 1
            
            for word, count in word_freq.items():
                if word in self.vocabulary_:
                    idx = self.vocabulary_[word]
                    # Enhanced TF calculation with normalization
                    tf = 0.5 + 0.5 * (count / max_freq)
                    idf = self.idf_[word]
                    vector[idx] = tf * idf
            
            features.append(vector)
        
        return np.array(features, dtype=np.float32)

# Enhanced Logistic Regression with better training
class EnhancedLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=2000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = 0
    
    def sigmoid(self, z):
        # Improved sigmoid with better numerical stability
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float32)
        
        # Add L2 regularization
        lambda_reg = 0.01
        
        for iteration in range(self.max_iter):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Calculate loss
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            
            # Gradients with regularization
            dw = (1/n_samples) * np.dot(X.T, (predictions - y)) + (lambda_reg/n_samples) * self.weights
            db = np.mean(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Early stopping if convergence
            if iteration > 0 and abs(loss - prev_loss) < self.tol:
                break
            prev_loss = loss
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        return (y_pred >= 0.5).astype(int)
    
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        prob = self.sigmoid(linear_pred)
        return np.column_stack([1 - prob, prob])

# Load CSV data with sampling to reduce memory usage
def load_csv_data(filename, sample_size=5000):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        
        # Sample data to reduce memory usage
        if len(rows) > sample_size:
            indices = np.random.choice(len(rows), sample_size, replace=False)
            rows = [rows[i] for i in indices]
        
        for row in rows:
            data.append(row)
    return data

# Global variables for model and vectorizer
model = None
vectorizer = None

# Load and train the model
def load_and_train_model():
    global model, vectorizer
    
    try:
        print("Loading data...")
        # Load data with sampling
        fake_data = load_csv_data("Fake.csv", sample_size=4000)
        true_data = load_csv_data("True.csv", sample_size=4000)
        
        print("Preprocessing data...")
        # Combine and shuffle data
        all_data = []
        
        # Add fake news (label 0)
        for row in fake_data:
            all_data.append({
                'text': wordopt(row['text']),
                'label': 0
            })
        
        # Add true news (label 1)
        for row in true_data:
            all_data.append({
                'text': wordopt(row['text']),
                'label': 1
            })
        
        # Shuffle data
        np.random.shuffle(all_data)
        
        # Separate features and labels
        texts = [item['text'] for item in all_data]
        labels = np.array([item['label'] for item in all_data], dtype=np.float32)
        
        print("Vectorizing text...")
        # Vectorize with enhanced features
        vectorizer = EnhancedTfidfVectorizer(max_features=2000, min_df=2, max_df=0.95)
        X = vectorizer.fit_transform(texts)
        
        print("Training model...")
        # Train enhanced model
        model = EnhancedLogisticRegression(learning_rate=0.01, max_iter=2000, tol=1e-4)
        model.fit(X, labels)
        
        print("Model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return False

# Initialize model on startup
print("Initializing Fake News Detector...")
if not load_and_train_model():
    print("Failed to load model. Please check your data files.")
    model = None
    vectorizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded. Please restart the application.'}), 500
        
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess the text
        processed_text = wordopt(text)
        
        # Vectorize
        text_vectorized = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        # Determine result
        result = "REAL" if prediction == 1 else "FAKE"
        confidence = float(probability[1] if prediction == 1 else probability[0])  # Convert to Python float
        
        return jsonify({
            'result': result,
            'confidence': round(confidence * 100, 2),
            'prediction': int(prediction)
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 