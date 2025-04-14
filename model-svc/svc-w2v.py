import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
nltk.download('stopwords')

def load_data(file_path='data/yelp.csv'):
    print(f'Loading {file_path}')
    df = pd.read_csv(file_path)
    return df['text'], df['stars']

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_text(text, remove_stopwords=True):
    """Tokenize and preprocess text"""
    tokens = simple_preprocess(text, deacc=True)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    return tokens

def train_word2vec(sentences, vector_size=100, window=5, min_count=5, workers=4):
    """Train Word2Vec model on the given sentences"""
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    return model

def document_to_vector(doc, word2vec_model):
    """Convert a document to a vector by averaging word vectors"""
    vectors = []
    for word in doc:
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])
    
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def main():
    X, y = load_data()

    # Preprocess all documents
    print("Preprocessing text...")
    processed_docs = [preprocess_text(text) for text in X]

    # Split data (must happen after preprocessing to maintain alignment)
    X_train, X_test, y_train, y_test = split_data(processed_docs, y)
    print(f'Data split: Train size {len(X_train)}, Test size {len(X_test)}')

    # Train Word2Vec model on the training data
    w2v_model = train_word2vec(X_train)

    # Convert documents to vectors
    print("Converting documents to vectors...")
    X_train_vectors = np.array([document_to_vector(doc, w2v_model) for doc in X_train])
    X_test_vectors = np.array([document_to_vector(doc, w2v_model) for doc in X_test])

    # C values to try
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    results = []

    # iterate C values
    for c in C_values:
        print(f"\nTraining and evaluating with C={c}")
        
        # Initialize and train model
        model = LinearSVC(
            C=c, 
            class_weight='balanced', 
            random_state=42, 
            max_iter=50000
        )
        model.fit(X_train_vectors, y_train)
        
        # Predict on test set
        y_test_pred = model.predict(X_test_vectors)

        # Calculate metrics
        acc = accuracy_score(y_test, y_test_pred)
        f1_macro = f1_score(y_test, y_test_pred, average='macro')
        f1_micro = f1_score(y_test, y_test_pred, average='micro')
        f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

        # Store results
        results.append({
            'C': c,
            'Accuracy': acc,
            'F1 Macro': f1_macro,
            'F1 Micro': f1_micro,
            'F1 Weighted': f1_weighted
        })

        print(f"Test Set Results for C={c} [LinearSVC with Word2Vec]:")
        print(f"Accuracy     : {acc:.4f}")
        print(f"F1 Macro     : {f1_macro:.4f}")
        print(f"F1 Micro     : {f1_micro:.4f}")
        print(f"F1 Weighted  : {f1_weighted:.4f}")

    # Find best performing model
    best_model = max(results, key=lambda x: x['F1 Macro'])
    print("\nBest performing model:")
    print(f"C Value      : {best_model['C']}")
    print(f"Accuracy     : {best_model['Accuracy']:.4f}")
    print(f"F1 Macro     : {best_model['F1 Macro']:.4f}")
    print(f"F1 Micro     : {best_model['F1 Micro']:.4f}")
    print(f"F1 Weighted  : {best_model['F1 Weighted']:.4f}")

if __name__ == "__main__":
    main()