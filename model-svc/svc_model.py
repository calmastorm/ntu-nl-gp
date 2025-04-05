import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
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

    # bow fit transform
    vectorizer = CountVectorizer(stop_words='english')
    X_train_bow = vectorizer.fit_transform(X_train)

    # C
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]

    # iterate C
    for c in C_values:
        model = LinearSVC(C=c, class_weight='balanced', random_state=42, max_iter=50000)

        y_train_pred = cross_val_predict(model, X_train_bow, y_train, cv=5)

        acc = accuracy_score(y_train, y_train_pred)
        f1_macro = f1_score(y_train, y_train_pred, average='macro')
        f1_micro = f1_score(y_train, y_train_pred, average='micro')
        f1_weighted = f1_score(y_train, y_train_pred, average='weighted')

        print(f"\nCross-Validation Results for C={c} [LinearSVC]:")
        print(f"Accuracy     : {acc:.4f}")
        print(f"F1 Macro     : {f1_macro:.4f}")
        print(f"F1 Micro     : {f1_micro:.4f}")
        print(f"F1 Weighted  : {f1_weighted:.4f}")

if __name__ == "__main__":
    main()
