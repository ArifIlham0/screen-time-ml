import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('data/weather_history.csv')

df.columns = ['date', 'summary', 'precip_type', 'temp', 'apparent_temp', 'humidity',
              'wind_speed', 'wind_bearing', 'visibility', 'pressure', 'cloud_cover', 'daily_summary']

df['date'] = pd.to_datetime(df['date'], utc=True)

df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
df.dropna(subset=['temp'], inplace=True)

df = df.sort_values('date')

daily_df = df.groupby(df['date'].dt.date)['temp'].mean().reset_index()
daily_df.columns = ['date', 'avg_temp']

for i in range(1, 4):
    daily_df[f'lag_{i}'] = daily_df['avg_temp'].shift(i)

daily_df.dropna(inplace=True)

features = ['lag_1', 'lag_2', 'lag_3']
X = daily_df[features]
y = daily_df['avg_temp']
X_train, y_train = X[:-3], y[:-3]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

last_known = X.iloc[-1].values.reshape(1, -1)
predictions = []
for i in range(3):
    pred = model.predict(last_known)[0]
    predictions.append(pred)
    last_known = np.roll(last_known, -1)
    last_known[0, -1] = pred

print("Prediksi suhu rata-rata untuk 3 hari ke depan:")
for i, p in enumerate(predictions, 1):
    print(f"Hari ke-{i}: {p:.2f} °C")

actual_days = daily_df['avg_temp'][-10:].tolist()
predicted_days = predictions
days = list(range(1, len(actual_days) + len(predicted_days) + 1))

plt.plot(days[:len(actual_days)], actual_days, marker='o', label='Data Aktual')
plt.plot(days[len(actual_days):], predicted_days, marker='x', label='Prediksi', color='red')
plt.xlabel('Hari')
plt.ylabel('Suhu Rata-rata Harian (°C)')
plt.title('Prediksi Suhu 3 Hari ke Depan')
plt.legend()
plt.show()
