import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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

    # TF-IDF fit transform
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)  # Transform test data

    # C values to evaluate
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]

    # Store results for comparison
    results = []

    # iterate through C values
    for c in C_values:
        print(f"\nTraining and evaluating with C={c}")
        
        # Initialize and train model
        model = LinearSVC(
            C=c, 
            class_weight='balanced', 
            random_state=42, 
            max_iter=50000
        )
        model.fit(X_train_tfidf, y_train)
        
        # Predict on test set
        y_test_pred = model.predict(X_test_tfidf)

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

        print(f"Test Set Results for C={c} [LinearSVC with TF-IDF]:")
        print(f"Accuracy     : {acc:.4f}")
        print(f"F1 Macro     : {f1_macro:.4f}")
        print(f"F1 Micro     : {f1_micro:.4f}")
        print(f"F1 Weighted  : {f1_weighted:.4f}")

    # Optionally: Find and print the best performing C value
    best_result = max(results, key=lambda x: x['F1 Macro'])  # Using F1 Macro as selection criteria
    print("\nBest performing model:")
    print(f"C Value      : {best_result['C']}")
    print(f"Accuracy     : {best_result['Accuracy']:.4f}")
    print(f"F1 Macro     : {best_result['F1 Macro']:.4f}")
    print(f"F1 Micro     : {best_result['F1 Micro']:.4f}")
    print(f"F1 Weighted  : {best_result['F1 Weighted']:.4f}")

if __name__ == "__main__":
    main()