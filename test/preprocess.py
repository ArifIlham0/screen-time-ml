import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load raw data
df = pd.read_csv('data/mobile_usage_merged.csv')

# Buat label kategori penggunaan secara manual
def categorize(row):
    if row['Daily_Screen_Time_Hours'] < 3:
        return 'Rendah'
    elif row['Daily_Screen_Time_Hours'] < 6:
        return 'Sedang'
    else:
        return 'Tinggi'

df['kategori_penggunaan'] = df.apply(categorize, axis=1)

# Simpan hasil preprocessing
df.to_csv('data/processed_data.csv', index=False)
print("Preprocessing selesai. Data disimpan ke data/processed_data.csv")
