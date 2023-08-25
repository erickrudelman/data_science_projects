import yfinance as yf
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import timedelta

st.write("""
# Stock Price App

Visualize the stock closing price, volume, and perform basic prediction of selected stocks!

""")

# Input ticker symbols
selected_tickers = st.multiselect('Select Ticker Symbols', ['GOOGL', 'AAPL', 'AMZN', 'MSFT'], default=['GOOGL'])

# Fetch data for selected tickers
ticker_data = {ticker: yf.Ticker(ticker).history(period='1d', start='2010-5-21', end='2023-8-25') for ticker in selected_tickers}

# Plot closing price and volume for selected tickers
st.write('## Closing Price and Volume Over Time')
for ticker, data in ticker_data.items():
    st.subheader(f'Closing Price and Volume - {ticker}')
    st.line_chart(data[['Close', 'Volume']], use_container_width=True)

# Comparative Analysis
st.write('## Comparative Analysis')
data_to_compare = [data['Close'] for ticker, data in ticker_data.items()]
df_comp = pd.concat(data_to_compare, axis=1)
df_comp.columns = selected_tickers

st.line_chart(df_comp, use_container_width=True)

# Prediction Model
st.write('## Stock Price Prediction')
selected_ticker = st.selectbox('Select Ticker Symbol for Prediction', selected_tickers)
data = ticker_data[selected_ticker]

# Prepare features (use lagged closing prices as features)
data['Lagged_Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

X = data[['Lagged_Close']]
y = data['Close']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the closing price for the next day
next_day_date = X_test.index[-1] + timedelta(days=1)
next_day_lagged_close = data.loc[X_test.index[-1], 'Close']
next_day_X = pd.DataFrame({'Lagged_Close': [next_day_lagged_close]}, index=[next_day_date])
next_day_pred = model.predict(next_day_X)[0]

# Display predictions and MSE
st.subheader(f'Predicted vs Actual Closing Price for {selected_ticker}')
y_pred = model.predict(X_test)  # Predictions for the test set
mse = mean_squared_error(y_test, y_pred)
prediction_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=X_test.index)
st.line_chart(prediction_df, use_container_width=True)
st.write(f'Mean Squared Error: {mse:.2f}')

