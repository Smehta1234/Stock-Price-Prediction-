import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MSE: {mse}, R-squared: {r2}")

    # Visualization
    plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
    plt.plot(y_test.index, predictions, label='Predicted Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()

    # Rotate x-axis labels and format date before calling plt.show()
    plt.gcf().autofmt_xdate()  # Rotate dates to avoid overlap
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format date to Year-Month

    # Display the plot
    plt.show()



if __name__ == '__main__':
    from src.utils import load_data_from_csv
    from src.data_preprocessing import preprocess_data
    from src.model_training import train_model

    # Load processed data
    data = load_data_from_csv('data/processed_stock_data.csv')
    processed_data = preprocess_data(data)

    # Load trained model
    model = joblib.load('models/linear_regression_model.pkl')

    # Evaluate model
    _, X_test, y_test = train_model(processed_data)
    evaluate_model(model, X_test, y_test)

