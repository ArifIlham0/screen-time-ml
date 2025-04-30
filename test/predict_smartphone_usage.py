import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
file_path = os.path.join(data_dir, 'mobile_usage_st.csv')
df = pd.read_csv(file_path)
features = df[['Daily_Screen_Time_Hours', 'Age', 'Total_App_Usage_Hours']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
df['cluster'] = clusters
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                               columns=[
                                   'Daily_Screen_Time_Hours', 
                                   'Age', 
                                   'Total_App_Usage_Hours'
                                ])
print("Centroid masing-masing cluster:")
print(cluster_centers)

cluster_centers['label'] = [''] * 3
sorted_centroids = cluster_centers.sort_values(by='Daily_Screen_Time_Hours')
labels = ['Rendah', 'Sedang', 'Tinggi']
for i, idx in enumerate(sorted_centroids.index):
    cluster_centers.at[idx, 'label'] = labels[i]

cluster_label_map = cluster_centers['label'].to_dict()
df['kategori_penggunaan'] = df['cluster'].map(cluster_label_map)

def prediksi_penggunaan(Daily_Screen_Time_Hours, Age, Total_App_Usage_Hours):
    input_data = pd.DataFrame([[Daily_Screen_Time_Hours, Age, Total_App_Usage_Hours]],
                              columns=['Daily_Screen_Time_Hours', 'Age', 'Total_App_Usage_Hours'])
    scaled_input = scaler.transform(input_data)
    cluster = kmeans.predict(scaled_input)[0]
    kategori = cluster_centers.loc[cluster, 'label']
    return kategori

print("=== Prediksi Input User ===")
user_screen_time = float(input("Masukkan screen time per hari (jam): "))
user_age = int(input("Masukkan usia: "))
user_app_usage = float(input("Masukkan total app usage (jam): "))

hasil = prediksi_penggunaan(user_screen_time, user_age, user_app_usage)
print(f"Kebiasaan penggunaan smartphone Anda termasuk: {hasil}")

sns.scatterplot(data=df, x='Daily_Screen_Time_Hours', y='Total_App_Usage_Hours', hue='kategori_penggunaan')
plt.title("Hasil Clustering Penggunaan Smartphone")
plt.xlabel("Daily Screen Time (jam)")
plt.ylabel("Total App Usage (jam)")
plt.show()