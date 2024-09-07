import pandas as pd
import numpy as np

from flask import Flask, render_template, redirect, url_for, request, jsonify
from pymongo import MongoClient
from binance.spot import Spot
from datetime import datetime, timedelta
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/')
db = client['crypto']
reddit_collection = db['reddit']
price_collection = db['btcusdt']

btc_keywords = "btc|bitcoin"
timedelta_query = 48
client = Spot()

def recursive_forecast(model, data, look_back, n_steps):
    forecast = []
    input_seq = data[-look_back:].reshape(1, look_back, data.shape[1])

    for _ in range(n_steps):
        pred = model.predict(input_seq)[0, 0]
        forecast.append(pred)
        
        # Update input sequence with the new prediction
        new_input = np.append(input_seq[0][1:], [[pred, 0]], axis=0)
        input_seq = new_input.reshape(1, look_back, data.shape[1])

    return forecast

@app.route("/")
def home():
    url_for('static', filename='style.css')

    #BTC query
    btc_query = {
        'open_time':{'$lt':datetime.utcnow().timestamp() * 1000, '$gt':( datetime.utcnow() - timedelta(hours=timedelta_query)).timestamp() * 1000},
    }
    df = pd.DataFrame(price_collection.find(btc_query))
    if df.empty:
        return 'DataFrame is empty!'
    else:
        df['x'] = df['open_time']
        df['h'] = pd.to_numeric(df['high'] )
        df['o'] = pd.to_numeric(df['open'] )    
        df['l'] = pd.to_numeric(df['low'] )
        df['c'] = pd.to_numeric(df['close'] )
        df['DateHour'] = pd.to_datetime(df['x'], unit='ms').dt.floor(freq='1min')


    #query reddit sentiment data
    reddit_query = {
        'created_utc':{'$lt':datetime.utcnow().timestamp(), '$gt':( datetime.utcnow() - timedelta(hours=timedelta_query)).timestamp()},
        "$or":[{"body": {'$regex' : btc_keywords, '$options': "i"}, "submission_title": {'$regex' : btc_keywords, '$options': "i"}}]
    }

    reddit_df = pd.DataFrame(reddit_collection.find(reddit_query))
    if reddit_df.empty:
        return 'DataFrame is empty!'
    else:
        reddit_df['created_utc'] = pd.to_datetime(reddit_df['created_utc'], unit='s')
        reddit_df['DateHour'] = reddit_df['created_utc'].dt.floor(freq='1min')
        print(reddit_df)
        reddit_df['sentiment'] = reddit_df.apply(lambda x: TextBlob(x['submission_title'] + " " + x['body']).sentiment.polarity, axis=1)
        print('After sentiment Scoring')
        print(reddit_df)
        reddit_df = reddit_df.groupby('DateHour').agg(
            sentiment=('sentiment', 'mean'),
            mention_nuber=('sentiment', 'count'),
        ).reset_index() 
    
    merged_df = pd.merge(df, reddit_df, on='DateHour', how='left').fillna(0)

    #train model

    # Scale combined data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_combined_data = scaler.fit_transform(merged_df[['c', 'sentiment']].values)

    # Create sequences for LSTM
    look_back = 12
    X, y = [], []
    for i in range(look_back, len(scaled_combined_data)):
        X.append(scaled_combined_data[i-look_back:i, :])
        y.append(scaled_combined_data[i, 0])
    X, y = np.array(X), np.array(y)


    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=25, batch_size=32)

    predicted_prices = model.predict(X)
    predicted_prices = scaler.inverse_transform(np.concatenate((predicted_prices, np.zeros((predicted_prices.shape[0], 1))), axis=1))[:, 0]
    actual_prices = merged_df[['c']].to_numpy()[look_back:, 0]
    print(actual_prices)
    print(predicted_prices)
    mse = mean_squared_error(actual_prices, predicted_prices)
    print(f"Mean Squared Error (MSE): {mse}")
    mape = mean_absolute_percentage_error(actual_prices, predicted_prices)
    print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

    #predict future price
    n_steps = 10  # Number of future steps to predict
    future_predictions = recursive_forecast(model, scaled_combined_data, look_back, n_steps)
    future_datetime = [merged_df['x'].iloc[-1] + i*5*60*1000 for i in range(1, n_steps + 1)]
    dummy_zero_array = np.zeros((len(future_predictions), scaled_combined_data.shape[1] - 1))
    predictions_full = np.concatenate((np.array(future_predictions).reshape(-1, 1), dummy_zero_array), axis=1)
    predictions_unscaled = scaler.inverse_transform(predictions_full)[:, 0]

    # Create DataFrame for predictions
    future_data = pd.DataFrame({
        'x': future_datetime,
        'y': predictions_unscaled,
    })

    #prepare data
    predict_data =  merged_df['x'][look_back:].reset_index(drop=True)
    predict_data = pd.DataFrame(predict_data, columns=['x'])
    predict_data = predict_data.assign(y=pd.Series(predicted_prices).values)
    # print(predict_data)
    coin_candlestick_data = merged_df[['x', 'o', 'h', 'l', 'c', 'volume']].to_dict(orient='records')
    coin_line_data = predict_data.to_dict(orient='records') 
    sentiment_data = merged_df[['x', 'sentiment']].rename(columns={"sentiment": "y"}).to_dict(orient='records')
    future_data = future_data.to_dict(orient='records')


    return render_template('main.html', chart_data=coin_candlestick_data, line_data=coin_line_data, future_data=future_data, sentiment_data=sentiment_data)


if __name__ == '__main__':
    app.run(debug=True)
