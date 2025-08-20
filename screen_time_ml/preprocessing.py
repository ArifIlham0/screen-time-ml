import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """
    Memuat data dari file CSV
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data berhasil dimuat dengan shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"File {file_path} tidak ditemukan!")
        return None
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        return None

def explore_data(data):
    """
    Eksplorasi data awal
    """
    print("\n=== EKSPLORASI DATA ===")
    print("\nInfo Dataset:")
    print(data.info())
    
    print("\nStatistik Deskriptif:")
    print(data.describe())
    
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    print("\nUnique Values per Column:")
    for col in data.columns:
        print(f"{col}: {data[col].nunique()} unique values")

def handle_missing_values(data):
    """
    Menangani missing values
    """
    print("\n=== MENANGANI MISSING VALUES ===")
    
    # Cek missing values
    missing_count = data.isnull().sum()
    if missing_count.sum() > 0:
        print("Missing values ditemukan:")
        print(missing_count[missing_count > 0])
        
        # Untuk kolom numerik: gunakan median
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
                print(f"Filled {col} dengan median: {median_val}")
        
        # Untuk kolom kategorikal: gunakan mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                mode_val = data[col].mode()[0]
                data[col].fillna(mode_val, inplace=True)
                print(f"Filled {col} dengan mode: {mode_val}")
    else:
        print("Tidak ada missing values ditemukan.")
    
    return data

# def handle_missing_values_visual(data):
#     """
#     Menangani missing values dan menampilkan data before/after secara visual
#     """
#     print("\n=== MENANGANI MISSING VALUES ===")
    
#     # Simpan data sebelum penanganan
#     data_before = data.copy()
#     missing_before = data_before.isnull().sum()
#     print("\nMissing values BEFORE (jumlah):")
#     print(missing_before)
#     print("\nData dengan missing values (BEFORE):")
#     print(data_before[data_before.isnull().any(axis=1)])

#     # Visualisasi missing values sebelum
#     plt.figure(figsize=(10, 4))
#     plt.bar(missing_before.index, missing_before.values, color='salmon')
#     plt.title('Missing Values per Column (Before)')
#     plt.ylabel('Jumlah Missing')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()

#     # Penanganan missing values
#     missing_count = missing_before
#     if missing_count.sum() > 0:
#         print("\nMissing values ditemukan:")
#         print(missing_count[missing_count > 0])
        
#         # Untuk kolom numerik: gunakan median
#         numeric_cols = data.select_dtypes(include=[np.number]).columns
#         for col in numeric_cols:
#             if data[col].isnull().sum() > 0:
#                 median_val = data[col].median()
#                 data[col] = data[col].fillna(median_val)
#                 print(f"Filled {col} dengan median: {median_val}")
        
#         # Untuk kolom kategorikal: gunakan mode
#         categorical_cols = data.select_dtypes(include=['object']).columns
#         for col in categorical_cols:
#             if data[col].isnull().sum() > 0:
#                 mode_val = data[col].mode()[0]
#                 data[col] = data[col].fillna(mode_val)
#                 print(f"Filled {col} dengan mode: {mode_val}")
#     else:
#         print("Tidak ada missing values ditemukan.")
    
#     # Simpan data setelah penanganan
#     data_after = data.copy()
#     missing_after = data_after.isnull().sum()
#     print("\nMissing values AFTER (jumlah):")
#     print(missing_after)
#     print("\nData dengan missing values (AFTER):")
#     print(data_after[data_after.isnull().any(axis=1)])

#     # Visualisasi missing values setelah
#     plt.figure(figsize=(10, 4))
#     plt.bar(missing_after.index, missing_after.values, color='skyblue')
#     plt.title('Missing Values per Column (After)')
#     plt.ylabel('Jumlah Missing')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()

#     # Tabel ringkas before/after
#     fig, ax = plt.subplots(figsize=(10, 4))
#     table_data = pd.DataFrame({'Before': missing_before, 'After': missing_after})
#     ax.axis('off')
#     tbl = ax.table(cellText=table_data.values,
#                    colLabels=table_data.columns,
#                    rowLabels=table_data.index,
#                    loc='center',
#                    cellLoc='center')
#     tbl.auto_set_font_size(False)
#     tbl.set_fontsize(10)
#     tbl.scale(1.2, 1.2)
#     plt.title('Missing Values Before & After')
#     plt.show()
    
#     return data

def remove_outliers(data, numeric_columns, method='iqr'):
    """
    Menghilangkan outliers menggunakan IQR method
    """
    print(f"\n=== MENGHILANGKAN OUTLIERS ({method.upper()}) ===")
    
    data_clean = data.copy()
    outliers_removed = 0
    
    for col in numeric_columns:
        Q1 = data_clean[col].quantile(0.25)
        Q3 = data_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)
        outliers_count = outliers_mask.sum()
        
        if outliers_count > 0:
            print(f"{col}: {outliers_count} outliers ditemukan")
            data_clean = data_clean[~outliers_mask]
            outliers_removed += outliers_count
    
    print(f"\nTotal outliers dihapus: {outliers_removed}")
    print(f"Data shape setelah cleaning: {data_clean.shape}")
    
    return data_clean

def select_numeric_features(data):
    """
    Memilih hanya kolom numerik untuk clustering
    """
    print("\n=== SELEKSI FITUR NUMERIK ===")
    
    # Kolom numerik yang akan digunakan
    numeric_features = [
        'App Usage Time (min/day)',
        'Screen On Time (hours/day)', 
        # 'Battery Drain (mAh/day)',
        'App Frequency',
        # 'Data Usage (MB/day)',
        'Age'
    ]
    
    # Filter kolom yang ada di dataset
    available_features = [col for col in numeric_features if col in data.columns]
    
    if len(available_features) != len(numeric_features):
        missing_features = set(numeric_features) - set(available_features)
        print(f"Warning: Kolom tidak ditemukan: {missing_features}")
    
    print(f"Fitur numerik yang akan digunakan: {available_features}")
    
    # Ekstrak data numerik
    numeric_data = data[available_features].copy()
    
    print(f"Shape data numerik: {numeric_data.shape}")
    print("\nStatistik fitur numerik:")
    print(numeric_data.describe())
    
    return numeric_data, available_features

# def select_numeric_features(data):
#     """
#     Memilih hanya kolom numerik untuk clustering dan visualisasi fitur sebelum/sesudah seleksi
#     """
#     print("\n=== SELEKSI FITUR NUMERIK ===")
    
#     # Semua fitur sebelum seleksi
#     all_features = list(data.columns)
#     print(f"Fitur sebelum seleksi ({len(all_features)}): {all_features}")

#     # Kolom numerik yang akan digunakan
#     numeric_features = [
#         'App Usage Time (min/day)',
#         'Screen On Time (hours/day)', 
#         # 'Battery Drain (mAh/day)',
#         'App Frequency',
#         # 'Data Usage (MB/day)',
#         'Age'
#     ]
    
#     # Filter kolom yang ada di dataset
#     available_features = [col for col in numeric_features if col in data.columns]
#     print(f"Fitur numerik yang akan digunakan ({len(available_features)}): {available_features}")

#     # Visualisasi fitur sebelum dan sesudah seleksi
#     fig, ax = plt.subplots(figsize=(12, 4))
#     ax.barh(range(len(all_features)), [1]*len(all_features), color='gray', label='Sebelum Seleksi')
#     ax.barh([all_features.index(f) for f in available_features], [1]*len(available_features), color='skyblue', label='Setelah Seleksi')
#     ax.set_yticks(range(len(all_features)))
#     ax.set_yticklabels(all_features)
#     ax.set_xlabel('Fitur')
#     ax.set_title('Visualisasi Fitur Sebelum dan Sesudah Seleksi')
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
    
#     # Ekstrak data numerik
#     numeric_data = data[available_features].copy()
    
#     print(f"Shape data numerik: {numeric_data.shape}")
#     print("\nStatistik fitur numerik:")
#     print(numeric_data.describe())
    
#     return numeric_data, available_features

def create_intensity_labels(data, numeric_features):
    """
    Membuat label intensitas berdasarkan kombinasi fitur
    (untuk referensi evaluasi - bukan untuk training)
    """
    print("\n=== MEMBUAT LABEL INTENSITAS REFERENSI ===")
    
    # Normalisasi fitur untuk scoring
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(data[numeric_features])
    
    # Hitung skor intensitas sebagai rata-rata semua fitur ternormalisasi
    intensity_score = np.mean(normalized_features, axis=1)
    
    # Buat kategori berdasarkan quintile
    labels = pd.qcut(intensity_score, q=5, labels=['Sangat Rendah', 'Rendah', 'Normal', 'Tinggi', 'Sangat Tinggi'])
    
    print("Distribusi label intensitas:")
    print(labels.value_counts().sort_index())
    
    return labels

# def visualize_label_distribution_before_after(data, numeric_features, labels):
#     """
#     Visualisasi distribusi skor intensitas sebelum dan sesudah pembuatan label
#     """
#     # Hitung skor intensitas sebelum labeling
#     scaler = MinMaxScaler()
#     normalized_features = scaler.fit_transform(data[numeric_features])
#     intensity_score = np.mean(normalized_features, axis=1)

#     # Visualisasi distribusi skor intensitas (before)
#     plt.figure(figsize=(12, 5))
#     plt.hist(intensity_score, bins=30, color='gray', alpha=0.7, edgecolor='black')
#     plt.title('Distribusi Skor Intensitas (Sebelum Labeling)')
#     plt.xlabel('Intensity Score')
#     plt.ylabel('Jumlah Sampel')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

#     # Visualisasi distribusi label (after)
#     plt.figure(figsize=(8, 5))
#     labels.value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
#     plt.title('Distribusi Label Intensitas (Setelah Labeling)')
#     plt.xlabel('Label Intensitas')
#     plt.ylabel('Jumlah Sampel')
#     plt.tight_layout()
#     plt.show()

def save_processed_data(data, numeric_data, labels, output_dir='data/processed_data'):
    """
    Menyimpan data yang sudah diproses
    """
    import os
    
    # Buat direktori jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Simpan data
    data.to_csv(f'{output_dir}/full_data_cleaned.csv', index=False)
    numeric_data.to_csv(f'{output_dir}/numeric_data.csv', index=False)
    
    # Simpan labels sebagai Series
    labels_df = pd.DataFrame({'intensity_label': labels})
    labels_df.to_csv(f'{output_dir}/intensity_labels.csv', index=False)
    
    print(f"\n=== DATA DISIMPAN DI FOLDER {output_dir} ===")
    print(f"1. full_data_cleaned.csv - Data lengkap yang sudah dibersihkan")
    print(f"2. numeric_data.csv - Data numerik untuk clustering")
    print(f"3. intensity_labels.csv - Label intensitas referensi")

def visualize_data_distribution(data, numeric_features):
    """
    Visualisasi distribusi data
    """
    print("\n=== MEMBUAT VISUALISASI DISTRIBUSI DATA ===")
    
    # Set style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(numeric_features):
        if i < len(axes):
            axes[i].hist(data[feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribusi {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
    
    # Hapus subplot kosong
    for i in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('data/processed_data/data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[numeric_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig('data/processed_data/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Fungsi utama untuk menjalankan preprocessing
    """
    print("=== PREPROCESSING DATA SMARTPHONE USAGE ===\n")
    
    # 1. Load data
    file_path = input("Masukkan path file CSV (atau tekan Enter untuk 'data/user_behavior_dataset.csv'): ").strip()
    if not file_path:
        file_path = 'data/user_behavior_dataset.csv'
    
    data = load_data(file_path)
    if data is None:
        return
    
    # 2. Eksplorasi data
    explore_data(data)
    
    # 3. Handle missing values
    data = handle_missing_values(data)
    
    # 4. Seleksi fitur numerik
    numeric_data, numeric_features = select_numeric_features(data)
    
    # 5. Remove outliers
    data_clean = remove_outliers(data, numeric_features)
    numeric_data_clean = data_clean[numeric_features]
    
    # 6. Buat label intensitas referensi
    intensity_labels = create_intensity_labels(data_clean, numeric_features)
    # visualize_label_distribution_before_after(data_clean, numeric_features, intensity_labels)
    
    # 7. Visualisasi
    visualize_data_distribution(numeric_data_clean, numeric_features)
    
    # 8. Simpan data
    save_processed_data(data_clean, numeric_data_clean, intensity_labels)
    
    print("\n=== PREPROCESSING SELESAI ===")
    print(f"Data siap untuk tahap normalisasi!")
    print(f"Jumlah sampel: {len(data_clean)}")
    print(f"Jumlah fitur: {len(numeric_features)}")

if __name__ == "__main__":
    main()