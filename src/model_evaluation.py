import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

def load_model(model_path: str):
    """Load the trained model from a pickle file."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_test_data(test_data_path: str):
    """Load the test dataset from CSV."""
    df = pd.read_csv(test_data_path)
    x_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values
    return x_test, y_test

def evaluate_model(model, x_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }


def save_metrics(metrics: dict, output_path: str):
    """Save the evaluation metrics to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def main():
    model = load_model('model.pkl')
    x_test, y_test = load_test_data('./data/features/test_bow.csv')
    metrics = evaluate_model(model, x_test, y_test)
    save_metrics(metrics, 'metrics.json')

if __name__ == '__main__':
    main()