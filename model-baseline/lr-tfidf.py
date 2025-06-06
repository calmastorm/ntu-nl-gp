import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def load_data(file_path='data/yelp.csv'):
    print(f'Loading {file_path}')
    df = pd.read_csv(file_path)
    return df['text'], df['stars']

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def main():
    X, y = load_data()

    # split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f'Data split: Train size {len(X_train)}, Test size {len(X_test)}.')

    # tf-idf fit transform
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # model
    model = LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight='balanced')
    
    # train the model
    model.fit(X_train_tfidf, y_train)
    
    # predict on test set
    y_test_pred = model.predict(X_test_tfidf)

    # acc + f1
    acc = accuracy_score(y_test, y_test_pred)
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    f1_micro = f1_score(y_test, y_test_pred, average='micro')
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

    print("\nTest Set Results (Logistic Regression + TF-IDF):")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Macro     : {f1_macro:.4f}")
    print(f"F1 Micro     : {f1_micro:.4f}")
    print(f"F1 Weighted  : {f1_weighted:.4f}")

if __name__ == "__main__":
    main()