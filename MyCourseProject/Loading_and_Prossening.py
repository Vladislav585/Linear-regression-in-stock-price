import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime


def get_vtb_data_moex():
    ticker = "VTBR"
    start_date = "2010-01-01"
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
    data_list = []
    columns = []
    start = 0

    while True:
        params = {'from': start_date, 'start': start, 'iss.meta': 'off'}
        response = requests.get(url, params=params)
        if response.status_code != 200: break
        json_data = response.json()
        if not columns: columns = json_data['history']['columns']
        rows = json_data['history']['data']
        if not rows: break
        data_list.extend(rows)
        start += 100
        if len(data_list) % 1000 == 0: print(f"Загружено {len(data_list)} строк...")

    df_full = pd.DataFrame(data_list, columns=columns)
    df = df_full[['TRADEDATE', 'CLOSE']].copy()
    df.columns = ['Date', 'Close']
    df.dropna(subset=['Close'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    df.loc[df['Date'] < '2024-07-01', 'Close'] *= 5000
    df = df.sort_values('Date')
    print(f"\nЗагружено {len(df)} торговых дней с учетом коррекции сплита.")
    return df


def prepare_features(df):
    feature_df = df.copy()

    feature_df = feature_df[feature_df['Close'] > 10]
    feature_df['Lag_1'] = feature_df['Close'].shift(1)
    feature_df['Lag_2'] = feature_df['Close'].shift(2)
    feature_df['Lag_3'] = feature_df['Close'].shift(3)
    feature_df['SMA_7'] = feature_df['Close'].rolling(window=7).mean()
    feature_df.dropna(inplace=True)
    print(f"Признаки успешно созданы. Размер итоговой таблицы: {feature_df.shape}")
    return feature_df


if __name__ == "__main__":
    vtb_data = get_vtb_data_moex()
    vtb_features = prepare_features(vtb_data)
    print("\nПример таблицы с признаками (последние 5 дней):")
    print(vtb_features[['Date', 'Close', 'Lag_1', 'Lag_2', 'Lag_3', 'SMA_7']].tail())
    if not vtb_data.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(vtb_data['Date'], vtb_data['Close'], color='#00416A')
        plt.title("Динамика акций ПАО ВТБ (Цены скорректированы с учетом сплита)")
        plt.xlabel("Дата")
        plt.ylabel("Цена (в пересчете на новые акции, руб)")
        plt.grid(True, alpha=0.3)
        plt.show()
    vtb_features.to_csv('vtb_prepared.csv', index=False)