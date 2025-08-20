import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import time
from sklearn.cluster import KMeans, DBSCAN
warnings.filterwarnings('ignore')

def load_clustering_results(data_dir='data/clustering_results'):
    """
    Memuat hasil clustering dan data ternormalisasi
    """
    try:
        # Load normalized data
        standardized_data = pd.read_csv('data/normalized_data/data_standardized.csv')
        minmax_data = pd.read_csv('data/normalized_data/data_minmax.csv')
        
        normalized_datasets = {
            'standardized': standardized_data,
            'minmax': minmax_data
        }
        
        # Load clustering labels
        kmeans_results = {}
        dbscan_results = {}
        
        for norm_method in ['standardized', 'minmax']:
            # Load K-means labels
            kmeans_labels = pd.read_csv(f'{data_dir}/kmeans_labels_{norm_method}.csv')
            kmeans_results[norm_method] = {
                'labels': kmeans_labels['kmeans_cluster'].values,
                'algorithm': 'kmeans'
            }
            
            # Load DBSCAN labels
            dbscan_labels = pd.read_csv(f'{data_dir}/dbscan_labels_{norm_method}.csv')
            dbscan_results[norm_method] = {
                'labels': dbscan_labels['dbscan_cluster'].values,
                'algorithm': 'dbscan'
            }
        
        print(f"Hasil clustering berhasil dimuat dari {data_dir}")
        return normalized_datasets, kmeans_results, dbscan_results
        
    except FileNotFoundError as e:
        print(f"Error: File tidak ditemukan - {e}")
        print("Pastikan sudah menjalankan clustering terlebih dahulu!")
        return None, None, None

def calculate_internal_metrics(data, labels, algorithm='kmeans'):
    """
    Menghitung internal evaluation metrics
    """
    metrics = {}
    
    # Filter noise points untuk DBSCAN
    if algorithm == 'dbscan':
        mask = labels != -1
        if mask.sum() < 2:
            return {
                'silhouette_score': np.nan,
                'davies_bouldin_score': np.nan,
                'n_clusters': 0,
                'n_noise': len(labels) - mask.sum()
            }
        data_filtered = data[mask]
        labels_filtered = labels[mask]
    else:
        data_filtered = data
        labels_filtered = labels
    
    # Silhouette Score (higher is better, range: -1 to 1)
    if len(set(labels_filtered)) > 1:
        metrics['silhouette_score'] = silhouette_score(data_filtered, labels_filtered)
    else:
        metrics['silhouette_score'] = np.nan
    
    # Davies-Bouldin Score (lower is better, range: 0 to infinity)
    if len(set(labels_filtered)) > 1:
        metrics['davies_bouldin_score'] = davies_bouldin_score(data_filtered, labels_filtered)
    else:
        metrics['davies_bouldin_score'] = np.nan
    
    # Additional metrics
    metrics['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
    if algorithm == 'dbscan':
        metrics['n_noise'] = (labels == -1).sum()
    else:
        metrics['n_noise'] = 0
    
    return metrics

def calculate_stability_metrics(data, labels1, labels2):
    """
    Menghitung stability metrics dengan membandingkan dua hasil clustering
    """
    # Adjusted Rand Index (higher is better, range: -1 to 1)
    ari = adjusted_rand_score(labels1, labels2)
    
    # Normalized Mutual Information (higher is better, range: 0 to 1)
    nmi = normalized_mutual_info_score(labels1, labels2)
    
    return {
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi
    }

def evaluate_clustering_quality(normalized_datasets, kmeans_results, dbscan_results):
    """
    Evaluasi kualitas clustering untuk semua kombinasi
    """
    print("=== EVALUASI KUALITAS CLUSTERING ===\n")
    
    evaluation_results = {}
    
    for norm_method, data in normalized_datasets.items():
        print(f"\n--- EVALUASI UNTUK DATA {norm_method.upper()} ---")
        
        # Evaluasi K-Means
        kmeans_labels = kmeans_results[norm_method]['labels']
        kmeans_metrics = calculate_internal_metrics(data.values, kmeans_labels, 'kmeans')
        
        print(f"\nK-Means Metrics:")
        print(f"  Silhouette Score: {kmeans_metrics['silhouette_score']:.4f}")
        print(f"  Davies-Bouldin Score: {kmeans_metrics['davies_bouldin_score']:.4f}")
        print(f"  Number of Clusters: {kmeans_metrics['n_clusters']}")
        
        # Evaluasi DBSCAN
        dbscan_labels = dbscan_results[norm_method]['labels']
        dbscan_metrics = calculate_internal_metrics(data.values, dbscan_labels, 'dbscan')
        
        print(f"\nDBSCAN Metrics:")
        if not np.isnan(dbscan_metrics['silhouette_score']):
            print(f"  Silhouette Score: {dbscan_metrics['silhouette_score']:.4f}")
            print(f"  Davies-Bouldin Score: {dbscan_metrics['davies_bouldin_score']:.4f}")
        else:
            print(f"  Metrics tidak dapat dihitung (cluster tidak memadai)")
        print(f"  Number of Clusters: {dbscan_metrics['n_clusters']}")
        print(f"  Number of Noise Points: {dbscan_metrics['n_noise']}")
        
        evaluation_results[norm_method] = {
            'kmeans': kmeans_metrics,
            'dbscan': dbscan_metrics
        }
    
    return evaluation_results

def map_clusters_to_categories(data, labels, feature_names, n_classes=5):
    """
    Memetakan cluster ke kategori intensitas berdasarkan rata-rata total fitur.
    """
    df = data.copy()
    df['cluster'] = labels

    # Hitung rata-rata fitur per cluster
    cluster_means = df.groupby('cluster')[feature_names].mean()
    
    # Tambahkan kolom total untuk representasi intensitas keseluruhan
    cluster_means['total'] = cluster_means.sum(axis=1)
    
    # Urutkan berdasarkan total
    sorted_clusters = cluster_means.sort_values('total').index.tolist()

    # Definisikan label kategori
    intensity_labels = ['Sangat Rendah', 'Lumayan Rendah', 'Normal', 'Lumayan Tinggi', 'Sangat Tinggi']
    cluster_to_label = {cluster_id: intensity_labels[i] 
                        for i, cluster_id in enumerate(sorted_clusters[:n_classes])}

    # Terapkan mapping
    df['kategori_intensitas'] = df['cluster'].map(cluster_to_label)

    # Hitung distribusi kategori
    kategori_counts = df['kategori_intensitas'].value_counts().sort_index()
    total = len(df)
    print("\n=== DISTRIBUSI KATEGORI INTENSITAS PENGGUNAAN SMARTPHONE ===")
    for label in intensity_labels:
        count = kategori_counts.get(label, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{label:17}: {count} data ({percentage:.2f}%)")

    return df[['cluster', 'kategori_intensitas']]


def compare_normalization_methods(evaluation_results):
    """
    Membandingkan performa clustering antara metode normalisasi
    """
    print(f"\n=== PERBANDINGAN METODE NORMALISASI ===")
    
    comparison_data = []
    
    for norm_method, results in evaluation_results.items():
        for algorithm, metrics in results.items():
            comparison_data.append({
                'Normalization': norm_method,
                'Algorithm': algorithm.upper(),
                'Silhouette_Score': metrics['silhouette_score'],
                'Davies_Bouldin_Score': metrics['davies_bouldin_score'],
                'N_Clusters': metrics['n_clusters'],
                'N_Noise': metrics['n_noise']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    return comparison_df

def compare_algorithm_performance(evaluation_results):
    """
    Membandingkan performa antara K-Means dan DBSCAN
    """
    print(f"\n=== PERBANDINGAN ALGORITMA CLUSTERING ===")
    
    # Analisis berdasarkan Silhouette Score
    silhouette_comparison = {}
    davies_bouldin_comparison = {}
    
    for norm_method, results in evaluation_results.items():
        kmeans_sil = results['kmeans']['silhouette_score']
        dbscan_sil = results['dbscan']['silhouette_score']
        
        kmeans_db = results['kmeans']['davies_bouldin_score']
        dbscan_db = results['dbscan']['davies_bouldin_score']
        
        if not np.isnan(dbscan_sil):
            silhouette_comparison[norm_method] = {
                'kmeans': kmeans_sil,
                'dbscan': dbscan_sil,
                'winner': 'K-Means' if kmeans_sil > dbscan_sil else 'DBSCAN'
            }
        
        if not np.isnan(dbscan_db):
            davies_bouldin_comparison[norm_method] = {
                'kmeans': kmeans_db,
                'dbscan': dbscan_db,
                'winner': 'K-Means' if kmeans_db < dbscan_db else 'DBSCAN'
            }
    
    print("Silhouette Score Comparison (higher is better):")
    for norm_method, comparison in silhouette_comparison.items():
        print(f"  {norm_method}: K-Means={comparison['kmeans']:.4f}, "
              f"DBSCAN={comparison['dbscan']:.4f} → Winner: {comparison['winner']}")
    
    print("\nDavies-Bouldin Score Comparison (lower is better):")
    for norm_method, comparison in davies_bouldin_comparison.items():
        print(f"  {norm_method}: K-Means={comparison['kmeans']:.4f}, "
              f"DBSCAN={comparison['dbscan']:.4f} → Winner: {comparison['winner']}")
    
    return silhouette_comparison, davies_bouldin_comparison

def evaluate_cluster_stability(normalized_datasets, kmeans_results, dbscan_results):
    """
    Evaluasi stabilitas clustering antara berbagai metode normalisasi
    """
    print(f"\n=== EVALUASI STABILITAS CLUSTERING ===")
    
    norm_methods = list(normalized_datasets.keys())
    
    if len(norm_methods) >= 2:
        # Bandingkan K-Means antara standardized dan minmax
        kmeans_stability = calculate_stability_metrics(
            normalized_datasets[norm_methods[0]].values,
            kmeans_results[norm_methods[0]]['labels'],
            kmeans_results[norm_methods[1]]['labels']
        )
        
        print(f"K-Means Stability (Standardized vs MinMax):")
        print(f"  Adjusted Rand Index: {kmeans_stability['adjusted_rand_index']:.4f}")
        print(f"  Normalized Mutual Info: {kmeans_stability['normalized_mutual_info']:.4f}")
        
        # Bandingkan DBSCAN antara standardized dan minmax
        dbscan_stability = calculate_stability_metrics(
            normalized_datasets[norm_methods[0]].values,
            dbscan_results[norm_methods[0]]['labels'],
            dbscan_results[norm_methods[1]]['labels']
        )
        
        print(f"\nDBSCAN Stability (Standardized vs MinMax):")
        print(f"  Adjusted Rand Index: {dbscan_stability['adjusted_rand_index']:.4f}")
        print(f"  Normalized Mutual Info: {dbscan_stability['normalized_mutual_info']:.4f}")
        
        return {
            'kmeans_stability': kmeans_stability,
            'dbscan_stability': dbscan_stability
        }
    
    return None

def create_evaluation_summary(evaluation_results, comparison_df, stability_results):
    """
    Membuat ringkasan evaluasi
    """
    print(f"\n=== RINGKASAN EVALUASI ===")
    
    summary = {}
    
    # Best performer berdasarkan Silhouette Score
    valid_results = comparison_df[comparison_df['Silhouette_Score'].notna()]
    if not valid_results.empty:
        best_silhouette = valid_results.loc[valid_results['Silhouette_Score'].idxmax()]
        summary['best_silhouette'] = {
            'method': f"{best_silhouette['Normalization']}_{best_silhouette['Algorithm']}",
            'score': best_silhouette['Silhouette_Score']
        }
        print(f"Best Silhouette Score: {best_silhouette['Normalization']}_{best_silhouette['Algorithm']} "
              f"({best_silhouette['Silhouette_Score']:.4f})")
    
    # Best performer berdasarkan Davies-Bouldin Score
    valid_db_results = comparison_df[comparison_df['Davies_Bouldin_Score'].notna()]
    if not valid_db_results.empty:
        best_davies_bouldin = valid_db_results.loc[valid_db_results['Davies_Bouldin_Score'].idxmin()]
        summary['best_davies_bouldin'] = {
            'method': f"{best_davies_bouldin['Normalization']}_{best_davies_bouldin['Algorithm']}",
            'score': best_davies_bouldin['Davies_Bouldin_Score']
        }
        print(f"Best Davies-Bouldin Score: {best_davies_bouldin['Normalization']}_{best_davies_bouldin['Algorithm']} "
              f"({best_davies_bouldin['Davies_Bouldin_Score']:.4f})")
    
    # Rekomendasi
    print(f"\n=== REKOMENDASI ===")
    if 'best_silhouette' in summary and 'best_davies_bouldin' in summary:
        if summary['best_silhouette']['method'] == summary['best_davies_bouldin']['method']:
            print(f"✓ Rekomendasi: {summary['best_silhouette']['method']}")
            print(f"  Metode ini konsisten terbaik pada kedua metrik utama")
        else:
            print(f"• Berdasarkan Silhouette Score: {summary['best_silhouette']['method']}")
            print(f"• Berdasarkan Davies-Bouldin Score: {summary['best_davies_bouldin']['method']}")
            print(f"• Pilih berdasarkan prioritas: clustering quality vs compactness")
    
    return summary

def save_evaluation_results(evaluation_results, comparison_df, stability_results, 
                           summary, output_dir='data/evaluation_results'):
    """
    Menyimpan hasil evaluasi
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Simpan comparison DataFrame
    comparison_df.to_csv(f'{output_dir}/clustering_comparison.csv', index=False)
    
    # Simpan detailed evaluation results
    detailed_results = []
    for norm_method, results in evaluation_results.items():
        for algorithm, metrics in results.items():
            row = {'normalization': norm_method, 'algorithm': algorithm}
            row.update(metrics)
            detailed_results.append(row)
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(f'{output_dir}/detailed_evaluation.csv', index=False)
    
    # Simpan stability results
    if stability_results:
        stability_df = pd.DataFrame(stability_results).T
        stability_df.to_csv(f'{output_dir}/stability_analysis.csv')
    
    # Simpan summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f'{output_dir}/evaluation_summary.csv', index=False)
    
    print(f"\n=== HASIL EVALUASI DISIMPAN DI FOLDER {output_dir} ===")


def show_sample_data_per_category(data, kategori_df, n_samples=5):
    """
    Menampilkan contoh data minimal n_samples dari tiap kategori intensitas
    """
    df_combined = pd.concat([data.reset_index(drop=True), kategori_df], axis=1)

    for kategori in df_combined['kategori_intensitas'].unique():
        if isinstance(kategori, str):
            sample = df_combined[df_combined['kategori_intensitas'] == kategori].head(n_samples)
            print(f"\n=== {kategori.upper()} ===")
            print(sample.to_string(index=False))
        else:
            # Lewati kategori yang bukan string (misal NaN)
            continue

def compare_execution_time(normalized_datasets):
    """
    Membandingkan waktu eksekusi K-Means dan DBSCAN untuk setiap metode normalisasi
    """
    print("\n=== PERBANDINGAN WAKTU EKSEKUSI CLUSTERING ===")
    results = []
    for norm_method, data in normalized_datasets.items():
        # K-Means
        start = time.time()
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(data.values)
        kmeans_time = time.time() - start

        # DBSCAN
        start = time.time()
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(data.values)
        dbscan_time = time.time() - start

        print(f"{norm_method.capitalize()} - KMeans: {kmeans_time:.4f}s, DBSCAN: {dbscan_time:.4f}s")
        results.append({
            'Normalization': norm_method,
            'KMeans_Time': kmeans_time,
            'DBSCAN_Time': dbscan_time
        })
    return pd.DataFrame(results)

def main():
    """
    Fungsi utama untuk evaluasi clustering
    """
    print("=== EVALUASI CLUSTERING SMARTPHONE USAGE DATA ===\n")
    
    # 1. Load hasil clustering
    normalized_datasets, kmeans_results, dbscan_results = load_clustering_results()
    if normalized_datasets is None:
        return
    
    # 2. Evaluasi kualitas clustering
    evaluation_results = evaluate_clustering_quality(
        normalized_datasets, kmeans_results, dbscan_results
    )
    
    # 3. Bandingkan metode normalisasi
    comparison_df = compare_normalization_methods(evaluation_results)
    
    # 4. Bandingkan performa algoritma
    silhouette_comp, davies_bouldin_comp = compare_algorithm_performance(evaluation_results)

    # 5. Evaluasi stabilitas clustering
    stability_results = evaluate_cluster_stability(
        normalized_datasets, kmeans_results, dbscan_results
    )

    # 6. Ringkasan evaluasi dan rekomendasi
    summary = create_evaluation_summary(
        evaluation_results, comparison_df, stability_results
    )

    # 7. Perbandingan waktu eksekusi
    execution_time_df = compare_execution_time(normalized_datasets)
    print("\n", execution_time_df)

    # 8. Tambahkan klasifikasi kategori dari hasil KMeans dan DBSCAN terbaik
    best_norm = summary['best_silhouette']['method'].split('_')[0]
    best_algo = summary['best_silhouette']['method'].split('_')[1].lower()
    best_data = normalized_datasets[best_norm]
    feature_names = best_data.columns.tolist()
    
    if best_algo == 'kmeans':
        best_labels = kmeans_results[best_norm]['labels']
        print(f"\n=== KLASIFIKASI KATEGORI DARI HASIL K-MEANS ({best_norm}) ===")
        kategori_df = map_clusters_to_categories(
            pd.DataFrame(best_data, columns=feature_names),
            best_labels,
            feature_names
        )
        show_sample_data_per_category(pd.DataFrame(best_data, columns=feature_names), kategori_df, n_samples=5)
    
    elif best_algo == 'dbscan':
        best_labels = dbscan_results[best_norm]['labels']
        n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        
        if n_clusters >= 2:
            print(f"\n=== KLASIFIKASI KATEGORI DARI HASIL DBSCAN ({best_norm}) ===")
            kategori_df = map_clusters_to_categories(
                pd.DataFrame(best_data, columns=feature_names),
                best_labels,
                feature_names
            )

            show_sample_data_per_category(pd.DataFrame(best_data, columns=feature_names), kategori_df, n_samples=5)
        else:
            print(f"\nDBSCAN tidak menghasilkan cukup cluster untuk klasifikasi kategori.")

    # 8. Simpan seluruh hasil evaluasi
    save_evaluation_results(
        evaluation_results, comparison_df, stability_results, summary
    )

if __name__ == "__main__":
    main()