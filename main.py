
from src.utils import fetch_stock_data, save_data_to_csv, load_data_from_csv
from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
import joblib

def main():
    # Fetch stock data
    data = fetch_stock_data('AAPL', '2015-01-01', '2023-01-01')
    save_data_to_csv(data, 'data/stock_data.csv')

    # Preprocess the data
    data = load_data_from_csv('data/stock_data.csv')
    processed_data = preprocess_data(data)

    # Train model
    model, X_test, y_test = train_model(processed_data)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
