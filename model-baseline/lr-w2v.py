import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data for tokenization
nltk.download('punkt')

def load_data(file_path='data/yelp.csv'):
    print(f'Loading {file_path}')
    df = pd.read_csv(file_path)
    return df['text'], df['stars']

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def tokenize_text(text_series):
    """Tokenize text using NLTK's word_tokenize"""
    return text_series.apply(lambda x: word_tokenize(x.lower()))

def train_word2vec(tokenized_text, vector_size=100, window=5, min_count=2, workers=4):
    """Train Word2Vec model on tokenized text"""
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=tokenized_text,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    return model

def document_vector(word2vec_model, doc):
    """Convert a document to a vector by averaging word vectors"""
    # Remove out-of-vocabulary words
    words = [word for word in doc if word in word2vec_model.wv]
    if len(words) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word2vec_model.wv[words], axis=0)

def main():
    X, y = load_data()

    # split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f'Data split: Train size {len(X_train)}, Test size {len(X_test)}.')

    # Tokenize text
    print("Tokenizing text...")
    X_train_tokens = tokenize_text(X_train)
    X_test_tokens = tokenize_text(X_test)

    # Train Word2Vec model
    w2v_model = train_word2vec(X_train_tokens)
    
    # Convert documents to vectors by averaging word vectors
    print("Creating document vectors...")
    X_train_w2v = np.array([document_vector(w2v_model, doc) for doc in X_train_tokens])
    X_test_w2v = np.array([document_vector(w2v_model, doc) for doc in X_test_tokens])

    # model
    model = LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight='balanced')

    # Train on full training set
    print("Training model on full training set...")
    model.fit(X_train_w2v, y_train)

    # Predict on test set
    y_test_pred = model.predict(X_test_w2v)

    # Calculate metrics
    acc = accuracy_score(y_test, y_test_pred)
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    f1_micro = f1_score(y_test, y_test_pred, average='micro')
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

    print("\nTest Set Results (Logistic Regression + Word2Vec):")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Macro     : {f1_macro:.4f}")
    print(f"F1 Micro     : {f1_micro:.4f}")
    print(f"F1 Weighted  : {f1_weighted:.4f}")

if __name__ == "__main__":
    main()