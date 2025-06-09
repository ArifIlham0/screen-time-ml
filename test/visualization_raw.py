import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/user_behavior_dataset.csv')

sns.countplot(x='User Behavior Class', data=df)
plt.title('Distribusi Kelas Perilaku Pengguna')
plt.xlabel('Kelas')
plt.ylabel('Jumlah')
plt.show()

plt.figure(figsize=(10, 8))
df_numeric = df.select_dtypes(include=['int64', 'float64'])

if 'User ID' in df_numeric.columns:
    df_numeric = df_numeric.drop('User ID', axis=1)

sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap Korelasi Fitur (Raw Dataset)')
plt.show()
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = pd.read_csv('data/processed_data.csv')

# sns.boxplot(data=df, x='kategori_penggunaan', y='Daily_Screen_Time_Hours')
# plt.title('Distribusi Screen Time per Kategori')
# plt.show()

# sns.pairplot(df, hue='kategori_penggunaan')
# plt.title("Pairplot Fitur terhadap Kategori")
# plt.show()
