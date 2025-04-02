import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def load_data(file_path='data/yelp.csv'):
    print(f'Loading {file_path}')
    df = pd.read_csv(file_path)
    return df['text'], df['stars']

def main():
    X, y = load_data()
    
    vectorizer = CountVectorizer(stop_words='english')
    X_bow = vectorizer.fit_transform(X)

    model = LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight='balanced')
    
    y_pred = cross_val_predict(model, X_bow, y, cv=5)

    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_micro = f1_score(y, y_pred, average='micro')
    f1_weighted = f1_score(y, y_pred, average='weighted')

    print("\nLogistic Regression (C=1.0) Evaluation:")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Macro     : {f1_macro:.4f}")
    print(f"F1 Micro     : {f1_micro:.4f}")
    print(f"F1 Weighted  : {f1_weighted:.4f}")

if __name__ == "__main__":
    main()
