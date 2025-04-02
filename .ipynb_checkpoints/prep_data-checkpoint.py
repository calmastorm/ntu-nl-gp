import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path='data/yelp.csv'):
    print(f'Loading {file_path}')
    df = pd.read_csv(file_path)
    X = df['text']
    y = df['stars']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    print(f'Data loaded: Train size {len(X_train)}, test size {len(X_test)}.')
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
