import numpy as np
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
import logging 

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    """
    Load test_size from params.yaml file.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['data_ingestion']['test_size']
        logger.debug('test size retrieved')
        return test_size
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except yaml.YAMLError as e:
        logger.error('yaml error')
        raise
    except Exception as e:
        logger.error('some error occured')
        raise

def read_data(url: str) -> pd.DataFrame:
    """
    Read CSV data from a given URL.
    """
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"Error: Failed to read data from URL: {url}")
        print(e)
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the tweet dataframe: drop tweet_id, filter sentiment classes, encode sentiment.
    """
    try:
        df = df.copy()
        df.drop(columns=['tweet_id'], inplace=True)
        df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        return df
    except KeyError as e:
        print(f"Error: Missing expected column in DataFrame: {e}")
        raise
    except Exception as e:
        print("Error: Failed while processing data.")
        print(e)
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """
    Save train and test data to CSV files in the specified path.
    """
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
        print(f"Data saved to {data_path}")
    except Exception as e:
        print("Error: Failed to save train/test data.")
        print(e)
        raise

def main():
    try:
        print("Starting data ingestion pipeline...")
        test_size = load_params('params.yaml')
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = process_data(df)

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        data_path = os.path.join('data', 'raw')
        save_data(data_path, train_data, test_data)

        print(f"Data ingestion completed. Train: {len(train_data)}, Test: {len(test_data)}")

    except Exception as e:
        print("Pipeline failed.")
        raise

if __name__ == '__main__':
    main()