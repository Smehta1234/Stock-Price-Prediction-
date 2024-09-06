import os
import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def save_data_to_csv(data, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the data to CSV
    data.to_csv(file_path)
def load_data_from_csv(file_path):
    return pd.read_csv(file_path, index_col='Date', parse_dates=True)

# Example Usage in main.py:
if __name__ == '__main__':
    from src.utils import fetch_stock_data, save_data_to_csv

    data = fetch_stock_data('AAPL', '2015-01-01', '2023-01-01')
    save_data_to_csv(data, 'data/stock_data.csv')
