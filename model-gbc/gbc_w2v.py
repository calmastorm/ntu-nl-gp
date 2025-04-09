import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')

# Load data
file_path = 'data/yelp.csv'
df = pd.read_csv(file_path)
X, y = df['text'], df['stars']

# Tokenize
X_tokens = [word_tokenize(text.lower()) for text in X]

# Train Word2Vec model
w2v_model = Word2Vec(sentences=X_tokens, vector_size=100, window=5, min_count=2, workers=4, sg=1)

# Function to convert a review into average word vector
def vectorize_text(tokens, model):
    valid_words = [word for word in tokens if word in model.wv]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[valid_words], axis=0)

# Vectorize all reviews
X_vectors = np.array([vectorize_text(tokens, w2v_model) for tokens in X_tokens])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# Classifier (you can swap in others like LogisticRegression, SVM, etc.)
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)

# 5-fold CV predictions
y_train_pred = cross_val_predict(model, X_train, y_train, cv=5, method='predict')

# Evaluation
acc = accuracy_score(y_train, y_train_pred)
f1_macro = f1_score(y_train, y_train_pred, average='macro')
f1_micro = f1_score(y_train, y_train_pred, average='micro')
f1_weighted = f1_score(y_train, y_train_pred, average='weighted')

print("\nCross-Validation Results (on training set) [GBC + Word2Vec]:")
print(f"Accuracy     : {acc:.4f}")
print(f"F1 Macro     : {f1_macro:.4f}")
print(f"F1 Micro     : {f1_micro:.4f}")
print(f"F1 Weighted  : {f1_weighted:.4f}")
