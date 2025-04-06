import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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
    gbc = GradientBoostingClassifier(random_state=42)

    # grid search 参数范围
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.05],
        'max_depth': [3, 5]
    }

    # Grid Search 设置
    grid_search = GridSearchCV(
        estimator=gbc,
        param_grid=param_grid,
        scoring='accuracy',  # 也可以换成 'f1_macro'
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    # 开始搜索
    grid_search.fit(X_train_bow, y_train)

    # 最佳模型
    best_model = grid_search.best_estimator_

    # 在训练集上评估最佳模型
    y_train_pred = best_model.predict(X_train_bow)

    acc = accuracy_score(y_train, y_train_pred)
    f1_macro = f1_score(y_train, y_train_pred, average='macro')
    f1_micro = f1_score(y_train, y_train_pred, average='micro')
    f1_weighted = f1_score(y_train, y_train_pred, average='weighted')

    print("\nBest Parameters Found by Grid Search:")
    print(grid_search.best_params_)
    print("\nCross-Validation Results (on training set) [Best GBC]:")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Macro     : {f1_macro:.4f}")
    print(f"F1 Micro     : {f1_micro:.4f}")
    print(f"F1 Weighted  : {f1_weighted:.4f}")

if __name__ == "__main__":
    main()
