
import pandas as pd

def preprocess_data(data):
    # Create moving averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

    # Drop rows with NaN values
    data.dropna(inplace=True)

    return data

# Example Usage:
if __name__ == '__main__':
    from src.utils import load_data_from_csv
    data = load_data_from_csv('data/stock_data.csv')
    processed_data = preprocess_data(data)
    processed_data.to_csv('data/processed_stock_data.csv')
