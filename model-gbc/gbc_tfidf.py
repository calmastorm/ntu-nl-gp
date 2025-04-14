import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

def main():
    file_path = 'data/yelp.csv'
    df = pd.read_csv(file_path)
    X, y = df['text'], df['stars']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)  # Transform test data using same vectorizer

    # Initialize and train the model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions on test set
    y_test_pred = model.predict(X_test_tfidf)

    # Calculate evaluation metrics
    acc = accuracy_score(y_test, y_test_pred)
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    f1_micro = f1_score(y_test, y_test_pred, average='micro')
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

    print("\nTest Set Results (Gradient Boosting Classifier + TF-IDF):")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Macro     : {f1_macro:.4f}")
    print(f"F1 Micro     : {f1_micro:.4f}")
    print(f"F1 Weighted  : {f1_weighted:.4f}")

if __name__ == "__main__":
    main()