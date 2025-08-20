import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import os

def load_processed_data(data_dir='data/processed_data'):
    """
    Memuat data yang sudah diproses dari tahap preprocessing
    """
    try:
        numeric_data = pd.read_csv(f'{data_dir}/numeric_data.csv')
        labels = pd.read_csv(f'{data_dir}/intensity_labels.csv')
        print(f"Data berhasil dimuat dari {data_dir}")
        print(f"Shape data numerik: {numeric_data.shape}")
        print(f"Shape labels: {labels.shape}")
        return numeric_data, labels
    except FileNotFoundError as e:
        print(f"Error: File tidak ditemukan - {e}")
        print("Pastikan sudah menjalankan preprocessing terlebih dahulu!")
        return None, None

def apply_standardization(data):
    """
    Menerapkan StandardScaler (Z-score normalization)
    """
    print("\n=== STANDARDIZATION (Z-SCORE) ===")
    
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    data_standardized = pd.DataFrame(data_standardized, columns=data.columns, index=data.index)
    
    print("Statistik setelah standardization:")
    print(f"Mean: {data_standardized.mean().round(4).tolist()}")
    print(f"Std: {data_standardized.std().round(4).tolist()}")
    
    return data_standardized, scaler

def apply_minmax_scaling(data):
    """
    Menerapkan MinMaxScaler (0-1 scaling)
    """
    print("\n=== MIN-MAX SCALING (0-1) ===")
    
    scaler = MinMaxScaler()
    data_minmax = scaler.fit_transform(data)
    data_minmax = pd.DataFrame(data_minmax, columns=data.columns, index=data.index)
    
    print("Statistik setelah min-max scaling:")
    print(f"Min: {data_minmax.min().round(4).tolist()}")
    print(f"Max: {data_minmax.max().round(4).tolist()}")
    
    return data_minmax, scaler

def compare_normalization_methods(original_data, standardized_data, minmax_data):
    """
    Membandingkan berbagai metode normalisasi
    """
    print("\n=== PERBANDINGAN METODE NORMALISASI ===")
    
    # Pilih satu fitur untuk perbandingan detail
    feature_sample = original_data.columns[0]
    
    comparison_stats = pd.DataFrame({
        'Original': [
            original_data[feature_sample].mean(),
            original_data[feature_sample].std(),
            original_data[feature_sample].min(),
            original_data[feature_sample].max(),
            original_data[feature_sample].skew()
        ],
        'Standardized': [
            standardized_data[feature_sample].mean(),
            standardized_data[feature_sample].std(),
            standardized_data[feature_sample].min(),
            standardized_data[feature_sample].max(),
            standardized_data[feature_sample].skew()
        ],
        'MinMax': [
            minmax_data[feature_sample].mean(),
            minmax_data[feature_sample].std(),
            minmax_data[feature_sample].min(),
            minmax_data[feature_sample].max(),
            minmax_data[feature_sample].skew()
        ]
    }, index=['Mean', 'Std', 'Min', 'Max', 'Skewness'])
    
    print(f"\nPerbandingan statistik untuk fitur '{feature_sample}':")
    print(comparison_stats.round(4))
    
    return comparison_stats

def visualize_normalization_comparison(original_data, standardized_data, minmax_data):
    """
    Visualisasi perbandingan metode normalisasi
    """
    print("\n=== MEMBUAT VISUALISASI PERBANDINGAN ===")
    
    # Pilih 2 fitur pertama untuk visualisasi
    features_to_plot = original_data.columns[:2]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    datasets = {
        'Original': original_data,
        'Standardized': standardized_data,
        'MinMax': minmax_data
    }
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, feature in enumerate(features_to_plot):
        for j, (name, data) in enumerate(datasets.items()):
            # Histogram
            axes[i, j].hist(data[feature], bins=30, alpha=0.7, 
                           color=colors[j], edgecolor='black')
            axes[i, j].set_title(f'{name}\n{feature}')
            axes[i, j].set_ylabel('Frequency')
            axes[i, j].grid(True, alpha=0.3)
            
            # Tambahkan statistik
            mean_val = data[feature].mean()
            std_val = data[feature].std()
            axes[i, j].axvline(mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.2f}')
            axes[i, j].legend()
    
    plt.tight_layout()
    plt.savefig('data/normalized_data/normalization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Box plot comparison
    fig, axes = plt.subplots(1, len(features_to_plot), figsize=(15, 6))
    if len(features_to_plot) == 1:
        axes = [axes]
    
    for i, feature in enumerate(features_to_plot):
        data_for_boxplot = []
        labels_for_boxplot = []
        
        for name, data in datasets.items():
            data_for_boxplot.append(data[feature])
            labels_for_boxplot.append(name)
        
        axes[i].boxplot(data_for_boxplot, labels=labels_for_boxplot)
        axes[i].set_title(f'Box Plot - {feature}')
        axes[i].set_ylabel('Values')
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('data/normalized_data/boxplot_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def select_best_normalization(original_data, standardized_data, minmax_data):
    """
    Memilih metode normalisasi terbaik berdasarkan karakteristik data
    """
    print("\n=== PEMILIHAN METODE NORMALISASI TERBAIK ===")
    
    # Analisis skewness untuk setiap metode
    skewness_analysis = pd.DataFrame({
        'Original': original_data.skew(),
        'Standardized': standardized_data.skew(),
        'MinMax': minmax_data.skew()
    })
    
    print("Analisis Skewness (nilai mendekati 0 lebih baik):")
    print(skewness_analysis.round(4))
    
    # Hitung rata-rata absolute skewness untuk setiap metode
    avg_skewness = skewness_analysis.abs().mean()
    print(f"\nRata-rata absolute skewness:")
    for method, skew_val in avg_skewness.items():
        print(f"{method}: {skew_val:.4f}")
    
    # Rekomendasi berdasarkan karakteristik
    print(f"\n=== REKOMENDASI ===")
    best_method = avg_skewness.idxmin()
    print(f"Metode dengan skewness terendah: {best_method}")
    
    print(f"\nPertimbangan pemilihan metode:")
    print(f"- StandardScaler: Baik jika data terdistribusi normal")
    print(f"- MinMaxScaler: Baik untuk algoritma yang sensitif terhadap scale")
    
    # Untuk clustering, biasanya StandardScaler atau MinMaxScaler lebih cocok
    recommended_methods = ['standardized', 'minmax']
    
    return recommended_methods, {
        'standardized': standardized_data,
        'minmax': minmax_data
    }

def save_normalized_data(normalized_datasets, labels, output_dir='data/normalized_data'):
    """
    Menyimpan data yang sudah dinormalisasi
    """
    # Buat direktori jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    for method_name, data in normalized_datasets.items():
        filename = f'{output_dir}/data_{method_name}.csv'
        data.to_csv(filename, index=False)
        print(f"Data {method_name} disimpan ke: {filename}")
    
    # Simpan ulang labels untuk konsistensi
    labels.to_csv(f'{output_dir}/intensity_labels.csv', index=False)
    
    print(f"\n=== DATA TERNORMALISASI DISIMPAN DI FOLDER {output_dir} ===")

def create_normalization_report(original_data, normalized_datasets, comparison_stats):
    """
    Membuat laporan lengkap normalisasi
    """
    print("\n=== LAPORAN NORMALISASI ===")
    
    report = []
    report.append("LAPORAN NORMALISASI DATA SMARTPHONE USAGE")
    report.append("=" * 50)
    report.append(f"Jumlah sampel: {len(original_data)}")
    report.append(f"Jumlah fitur: {len(original_data.columns)}")
    report.append(f"Fitur: {list(original_data.columns)}")
    report.append("")
    
    report.append("STATISTIK DATA ORIGINAL:")
    report.append(str(original_data.describe()))
    report.append("")
    
    report.append("PERBANDINGAN METODE NORMALISASI:")
    report.append(str(comparison_stats))
    report.append("")
    
    for method_name, data in normalized_datasets.items():
        report.append(f"STATISTIK DATA {method_name.upper()}:")
        report.append(str(data.describe()))
        report.append("")
    
    # Simpan laporan ke file
    with open('data/normalized_data/normalization_report.txt', 'w') as f:
        f.write('\n'.join(report))

    print("Laporan normalisasi disimpan ke: data/normalized_data/normalization_report.txt")

def main():
    """
    Fungsi utama untuk menjalankan normalisasi
    """
    print("=== NORMALISASI DATA SMARTPHONE USAGE ===\n")
    
    # 1. Load data yang sudah diproses
    numeric_data, labels = load_processed_data()
    if numeric_data is None:
        return
    
    print("\nData original:")
    print(numeric_data.describe())
    
    # 2. Terapkan berbagai metode normalisasi
    standardized_data, std_scaler = apply_standardization(numeric_data)
    minmax_data, minmax_scaler = apply_minmax_scaling(numeric_data)
    
    # 3. Bandingkan metode normalisasi
    comparison_stats = compare_normalization_methods(numeric_data, standardized_data, minmax_data)
    
    # 4. Visualisasi perbandingan
    visualize_normalization_comparison(numeric_data, standardized_data, minmax_data)
    
    # 5. Pilih metode terbaik
    recommended_methods, normalized_datasets = select_best_normalization(numeric_data, standardized_data, minmax_data)

    # 6. Simpan data ternormalisasi
    save_normalized_data(normalized_datasets, labels)
    
    # 7. Buat laporan
    create_normalization_report(numeric_data, normalized_datasets, comparison_stats)
    
    print(f"\n=== NORMALISASI SELESAI ===")
    print(f"Data ternormalisasi siap untuk clustering!")
    print(f"Metode yang direkomendasikan: {recommended_methods}")

if __name__ == "__main__":
    main()