import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
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

    # model
    model = GradientBoostingClassifier(
        n_estimators=200,    # 总共100棵小树
        learning_rate=0.1,   # 每棵树的贡献系数
        max_depth=5,         # 每棵树的最大深度
        random_state=42
    )

    # 5 fold cv
    y_train_pred = cross_val_predict(model, X_train_bow, y_train, cv=5, method='predict')

    # acc + f1
    acc = accuracy_score(y_train, y_train_pred)
    f1_macro = f1_score(y_train, y_train_pred, average='macro')
    f1_micro = f1_score(y_train, y_train_pred, average='micro')
    f1_weighted = f1_score(y_train, y_train_pred, average='weighted')

    print("\nCross-Validation Results (on training set) [GBC]:")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Macro     : {f1_macro:.4f}")
    print(f"F1 Micro     : {f1_micro:.4f}")
    print(f"F1 Weighted  : {f1_weighted:.4f}")

if __name__ == "__main__":
    main()
