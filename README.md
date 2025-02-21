Sales Forecasting Model: Implementation Documentation

1. Introduction

This document details the implementation of a sales forecasting model designed to predict future sales based on historical data. The model leverages time series analysis techniques and machine learning algorithms to provide accurate and actionable insights. We will primarily use Python with libraries like Pandas, Statsmodels, Prophet, and Scikit-learn.

2. Data Acquisition and Preprocessing

2.1. Data Sources

Internal Data: Obtain sales data from the company's POS (Point of Sale) system, CRM, or sales management platform. The data should include at least the following columns:

week: The week identifier (week number or date). We'll convert this to a datetime object.

sales: The total sales value for that week.

External Data (Optional): Consider supplementing internal data with relevant datasets from Kaggle, Hugging Face, or other sources. Look for datasets related to retail sales, time series forecasting, or economic indicators.

2.2. Data Loading and Inspection

import pandas as pd

# Load internal sales data
df = pd.read_csv("internal_sales_data.csv")

# Inspect the first few rows
print(df.head())

# Check data types and missing values
print(df.info())

# Describe basic statistics
print(df.describe())
Use code with caution.
Python
2.3. Data Preprocessing

# Convert 'week' to datetime and set as index
df['week'] = pd.to_datetime(df['week'])
df = df.set_index('week')

# Handle Missing Values (Example: Linear Interpolation)
df['sales'] = df['sales'].interpolate(method='linear')

# Outlier Detection (Example: Using Z-score)
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(df['sales']))
threshold = 3  # Adjust threshold as needed
outlier_indices = np.where(z_scores > threshold)[0]

# Outlier Treatment (Example: Replacing with median of neighboring values)
for i in outlier_indices:
    # Find a window of non-outlier neighbors
    window_size = 1
    while True:
      lower_bound = max(0, i - window_size)
      upper_bound = min(len(df), i + window_size + 1)
      neighbors = df['sales'][lower_bound:upper_bound]
      non_outlier_neighbors = neighbors[~np.isin(neighbors.index, df.index[outlier_indices])]
      if len(non_outlier_neighbors) > 0:
            df.loc[df.index[i], 'sales'] = non_outlier_neighbors.median()
            break
      window_size += 1
      if lower_bound == 0 and upper_bound == len(df): #no non-outlier
          df.loc[df.index[i], 'sales'] = df['sales'].median() #use all data median
          break
Use code with caution.
Python
2.4. Feature Engineering

# Lagged Features
df['sales_lag1'] = df['sales'].shift(1)
df['sales_lag2'] = df['sales'].shift(2)
# ... Add more lagged features as needed

# Rolling Window Statistics (Example: 4-week rolling average)
df['sales_rolling_mean_4'] = df['sales'].rolling(window=4).mean()
df['sales_rolling_std_4'] = df['sales'].rolling(window=4).std()

# Time-Based Features
df['week_of_year'] = df.index.isocalendar().week
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['year'] = df.index.year
df['dayofweek'] = df.index.dayofweek

# Dummy Variables (Example: Holidays)
holidays = pd.DataFrame({
    'holiday': ['NewYear', 'Christmas'],
    'ds': pd.to_datetime(['2023-01-01', '2023-12-25']), #add all holiday
    'lower_window': [-1, -2], # Days before
    'upper_window': [1, 1],  # Days after
})

# One-Hot Encode Holidays (if not using Prophet)
#  You would need to create a boolean column for each holiday,
#  and set it to 1 if the current date falls within the holiday's window.

# Drop rows with NaN values created by feature engineering
df = df.dropna()
print(df.head())
Use code with caution.
Python
3. Model Selection and Training

3.1. Train-Test Split

# Split data into training and testing sets (e.g., 80/20 split)
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]
Use code with caution.
Python
3.2. ARIMA/SARIMA Model

from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# Option 1: Auto ARIMA to find optimal parameters
model_auto = auto_arima(train_data['sales'],
                        seasonal=True, m=52,  # m is the seasonal period (52 for weekly data)
                        trace=True,  # Print search progress
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)  # Use stepwise algorithm for faster search
print(model_auto.summary())

# Option 2: Manually specify ARIMA/SARIMA parameters
#  Based on ACF/PACF plots or prior knowledge
p, d, q = 1, 1, 1  # Example ARIMA parameters
P, D, Q, m = 1, 1, 1, 52  # Example SARIMA parameters

model = SARIMAX(train_data['sales'],
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit()
print(results.summary())

# Make predictions on the test set
predictions = results.get_forecast(steps=len(test_data))
predicted_values = predictions.predicted_mean
confidence_intervals = predictions.conf_int()
Use code with caution.
Python
3.3. Prophet Model

from prophet import Prophet

# Prepare data for Prophet (ds and y columns)
train_data_prophet = train_data.reset_index().rename(columns={'week': 'ds', 'sales': 'y'})
test_data_prophet = test_data.reset_index().rename(columns={'week': 'ds', 'sales': 'y'})

# Initialize and fit the Prophet model
model = Prophet(holidays=holidays) # Use the holidays DataFrame
# Add other regressors if needed
# model.add_regressor('sales_lag1')
model.fit(train_data_prophet)

# Create a future DataFrame for predictions
future = model.make_future_dataframe(periods=len(test_data), freq='W') # W: weekly

# Add future values for regressors if used
# future['sales_lag1'] = ...

# Make predictions
forecast = model.predict(future)

# Extract predicted values
predicted_values_prophet = forecast['yhat'][-len(test_data):]
confidence_intervals_prophet = forecast[['yhat_lower', 'yhat_upper']][-len(test_data):]
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
Use code with caution.
Python
4. Model Evaluation

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Evaluate ARIMA/SARIMA
mae = mean_absolute_error(test_data['sales'], predicted_values)
mse = mean_squared_error(test_data['sales'], predicted_values)
rmse = np.sqrt(mse)
r2 = r2_score(test_data['sales'], predicted_values)

print(f"ARIMA/SARIMA - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R-squared: {r2}")

# Evaluate Prophet
mae_prophet = mean_absolute_error(test_data['sales'], predicted_values_prophet)
mse_prophet = mean_squared_error(test_data['sales'], predicted_values_prophet)
rmse_prophet = np.sqrt(mse_prophet)
r2_prophet = r2_score(test_data['sales'], predicted_values_prophet)

print(f"Prophet - MAE: {mae_prophet}, MSE: {mse_prophet}, RMSE: {rmse_prophet}, R-squared: {r2_prophet}")

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['sales'], label='Actual')
plt.plot(test_data.index, predicted_values, label='ARIMA/SARIMA Predicted')
plt.plot(test_data.index, predicted_values_prophet, label='Prophet Predicted')

#For ARIMA
plt.fill_between(test_data.index,
                 confidence_intervals.iloc[:, 0],
                 confidence_intervals.iloc[:, 1], color='pink', alpha=0.5, label = "ARIMA Confidence Interval")

#For Prophet
plt.fill_between(test_data.index,
                 confidence_intervals_prophet['yhat_lower'],
                 confidence_intervals_prophet['yhat_upper'], color='skyblue', alpha=0.5 , label = "Prophet Confidence Interval")

plt.legend()
plt.title('Sales Forecast: Actual vs. Predicted')
plt.xlabel('Week')
plt.ylabel('Sales')
plt.show()
Use code with caution.
Python
5. Model Deployment

5.1. Model Saving

import pickle

# Save ARIMA/SARIMA model
results.save('arima_model.pkl')

# Save Prophet model
with open('prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)
Use code with caution.
Python
5.2. Model Loading

# Load ARIMA/SARIMA model
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
loaded_arima_model = SARIMAXResults.load('arima_model.pkl')

# Load Prophet model
with open('prophet_model.pkl', 'rb') as f:
    loaded_prophet_model = pickle.load(f)
Use code with caution.
Python
5.3 API with Flask/FastAPI

Create a simple API endpoint to serve predictions. Here's a basic example using Flask:

from flask import Flask, request, jsonify
import pandas as pd
# ... (load your model here - either ARIMA or Prophet) ...
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
# Load ARIMA/SARIMA model
loaded_arima_model = SARIMAXResults.load('arima_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Assuming input is a list of dates for which to predict
        dates = pd.to_datetime(data['dates'])

        # Create a DataFrame for the input dates (ARIMA example)
        future_df = pd.DataFrame(index=dates)

        # Make predictions using the loaded ARIMA model
        predictions = loaded_arima_model.get_forecast(steps=len(future_df))
        predicted_sales = predictions.predicted_mean.values.tolist()

        return jsonify({'predictions': predicted_sales})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) # Or any port you prefer
Use code with caution.
Python
5.4 Streamlit for Visualization
You could use Streamlit:

import streamlit as st
import pandas as pd
#...
# Assuming loaded_arima_model or loaded_prophet_model is available

st.title('Sales Forecasting Dashboard')

# Date input
start_date = st.date_input("Start Date", pd.to_datetime('today'))
num_weeks = st.number_input("Number of Weeks to Forecast", min_value=1, value=4)
end_date = start_date + pd.Timedelta(weeks=num_weeks)

# Generate dates for prediction
dates = pd.date_range(start=start_date, end=end_date, freq='W')

# Make predictions
# Example using loaded ARIMA model.  Adapt for Prophet as needed.
future_df = pd.DataFrame(index=dates)
predictions = loaded_arima_model.get_forecast(steps=len(future_df))
predicted_sales = predictions.predicted_mean
confidence_intervals = predictions.conf_int()

# Display predictions
st.subheader("Sales Forecast")
forecast_df = pd.DataFrame({'Date': dates, 'Predicted Sales': predicted_sales})
st.line_chart(forecast_df.set_index('Date'))

# Display confidence intervals
st.subheader("Confidence Intervals")
st.write(confidence_intervals)

# ... (add more visualizations and analysis as needed) ...
Use code with caution.
Python
6. Model Monitoring and Retraining

Performance Monitoring: Continuously monitor the model's performance using metrics like MAE, RMSE, and R-squared. Track these metrics over time and set up alerts for significant deviations from expected performance.

Retraining: Establish a schedule for retraining the model (e.g., weekly, monthly) or trigger retraining when performance drops below a predefined threshold. Use the latest available data for retraining. Consider using a rolling window approach, where you always train on a fixed-size window of the most recent data.

A/B test: A/B test new model with old model for best result.

7. Conclusion
This documentation provides a starting point and can be further enhanced with detail based on new data or information.
