import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_model(data):
    # Define the features and target
    X = data[['Close', 'SMA_20', 'EMA_20']]
    y = data['Close'].shift(-1)  # Predict next day's closing price

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, shuffle=False)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ensure the models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the model
    joblib.dump(model, 'models/linear_regression_model.pkl')

    return model, X_test, y_test



if __name__ == '__main__':
    from src.utils import load_data_from_csv
    from src.data_preprocessing import preprocess_data

    data = load_data_from_csv('data/processed_stock_data.csv')
    processed_data = preprocess_data(data)
    train_model(processed_data)
