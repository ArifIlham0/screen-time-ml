import joblib
import pandas as pd

# Load model dan scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Mapping label
label_map = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}

print("=== Prediksi Penggunaan Smartphone ===")
screen_time = float(input("Masukkan screen time per hari (jam): "))
age = int(input("Masukkan usia: "))
app_usage = float(input("Masukkan total penggunaan aplikasi (jam): "))

# Buat dataframe input
input_df = pd.DataFrame([[screen_time, age, app_usage]],
                        columns=['Daily_Screen_Time_Hours', 'Age', 'Total_App_Usage_Hours'])

scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]

print(f"Kebiasaan penggunaan smartphone Anda termasuk: {label_map[prediction]}")
