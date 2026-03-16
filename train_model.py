import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Daten laden (Variabel bis Ende 2022)
symbol = "BTC-USD"
train_end = "2022-12-31"
data = yf.download(symbol, start="2015-01-01", end=train_end)
prices = data['Close'].values.reshape(-1, 1)

# 2. Skalieren (Wichtig für neuronale Netze)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# 3. Daten in Fenster schneiden (z.B. 60 Tage schauen -> 1 Tag vorhersagen)
prediction_days = 60
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 4. Modell bauen
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1) # Vorhersage des nächsten Preises
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Modell und Scaler speichern
model.save('btc_model.h5')
import joblib
joblib.dump(scaler, 'scaler.gz')
print("Training abgeschlossen und Modell gespeichert.")
