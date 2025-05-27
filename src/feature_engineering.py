import numpy as np
import pandas as pd

import os
import yaml

from sklearn.feature_extraction.text import CountVectorizer

def load_params(params_path: str) -> int:
    """Load max_features parameter from YAML config."""
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params['features_engineering']['max_features']

#fretch the data from data/processed
def load_processed_data(train_path: str, test_path: str):
    """Load train and test data from processed CSV files."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df.fillna('', inplace=True)
    test_df.fillna('', inplace=True)

    return train_df, test_df


def apply_bow(train_texts, test_texts, max_features: int):
    """Apply CountVectorizer (BoW) to training and testing data."""
    vectorizer = CountVectorizer(max_features=max_features)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    return x_train, x_test

def create_feature_dfs(x_train_bow, y_train, x_test_bow, y_test):
    """Convert sparse matrices to DataFrames with labels."""
    train_df = pd.DataFrame(x_train_bow.toarray())
    train_df['label'] = y_train

    test_df = pd.DataFrame(x_test_bow.toarray())
    test_df['label'] = y_test

    return train_df, test_df

def save_feature_data(train_df, test_df, output_path: str):
    """Save the feature DataFrames as CSV files."""
    os.makedirs(output_path, exist_ok=True)
    train_df.to_csv(os.path.join(output_path, 'train_bow.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test_bow.csv'), index=False)

def main():
    max_features = load_params('params.yaml')

    train_data, test_data = load_processed_data(
        './data/processed/train_processed.csv',
        './data/processed/test_processed.csv'
    )

    x_train = train_data['content'].values
    y_train = train_data['sentiment'].values
    x_test = test_data['content'].values
    y_test = test_data['sentiment'].values

    x_train_bow, x_test_bow = apply_bow(x_train, x_test, max_features)

    train_df, test_df = create_feature_dfs(x_train_bow, y_train, x_test_bow, y_test)

    save_feature_data(train_df, test_df, './data/features')

if __name__ == "__main__":
    main()

