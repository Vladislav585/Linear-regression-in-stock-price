import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import random
import tensorflow as tf

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)
def load_data():
    df = pd.read_csv("vtb_prepared.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

def sample_splitting(df):
    x = df[['Lag_1', 'Lag_2', 'Lag_3', 'SMA_7']].values
    y = df[['Close']]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(x)
    Y_scaled = scaler_y.fit_transform(y)

    split_ind = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:split_ind]
    X_test = X_scaled[split_ind:]

    y_train = Y_scaled[:split_ind]
    y_test = Y_scaled[split_ind:]

    model = Sequential()
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.add(Dense(128, activation="relu", input_shape=(4,)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add((Dense(1)))
    model.compile(optimizer="adam", loss="mse")

    neuro = model.fit(X_train, y_train, batch_size=64, epochs=30, validation_split=0.1, verbose=1, callbacks=[early_stop])

    scores = model.evaluate(X_test, y_test)
    print(f'MSE:{scores}')
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"Коэффициент детерминации: {r2:.4f}")
    predictions = model.predict(X_test)
    predictions = scaler_y.inverse_transform(predictions)
    actual_prices = scaler_y.inverse_transform(y_test)
    return predictions, actual_prices, model, scaler_y, X_test

def graph(actual_prices, predictions, future_predictions=None):
    plt.figure(figsize=(12,6))
    plt.plot(actual_prices, label="Реальная цена ВТБ", color="blue")
    plt.plot(predictions, label="Предсказанная цена ВТБ", color="orange", linestyle="--")

    if future_predictions is not None:
        x_start = len(actual_prices)
        x_range = range(x_start, x_start + len(future_predictions))
        plt.plot(x_range, future_predictions, label="Прогноз на месяц вперед", color="red", linewidth=2)

    plt.title("Сравнение реальных и предсказанных котировок цен ВТБ")
    plt.xlabel("Торговые дни")
    plt.ylabel("Цена акции (руб.)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def forecast_future(model, last_real_data, scaler_y, n_days=22):
    future_predictions = []
    current_features = last_real_data[-1].reshape(1, 4)

    for i in range(n_days):
        pred_scaled = model.predict(current_features, verbose=0)
        future_predictions.append(pred_scaled[0, 0])

        new_features = np.zeros((1, 4))
        new_features[0, 0] = pred_scaled[0, 0]
        new_features[0, 1] = current_features[0, 0]
        new_features[0, 2] = current_features[0, 1]
        new_features[0, 3] = np.mean(new_features[0, :3])
        current_features = new_features

    future_rubles = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_rubles