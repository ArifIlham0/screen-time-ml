import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed_data.csv')

# Visualisasi distribusi screen time berdasarkan kategori
sns.boxplot(data=df, x='kategori_penggunaan', y='Daily_Screen_Time_Hours')
plt.title('Distribusi Screen Time per Kategori')
plt.show()

# Pairplot untuk melihat sebaran data
sns.pairplot(df, hue='kategori_penggunaan')
plt.title("Pairplot Fitur terhadap Kategori")
plt.show()
