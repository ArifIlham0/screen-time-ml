import os
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')

def load_normalized_data(data_dir='data/normalized_data'):
    """
    Memuat data yang sudah dinormalisasi
    """
    try:
        # Load data ternormalisasi
        standardized_data = pd.read_csv(f'{data_dir}/data_standardized.csv')
        minmax_data = pd.read_csv(f'{data_dir}/data_minmax.csv')
        labels = pd.read_csv(f'{data_dir}/intensity_labels.csv')
        
        print(f"Data berhasil dimuat dari {data_dir}")
        print(f"Shape data standardized: {standardized_data.shape}")
        print(f"Shape data minmax: {minmax_data.shape}")
        print(f"Shape labels: {labels.shape}")
        
        return {
            'standardized': standardized_data,
            'minmax': minmax_data
        }, labels
        
    except FileNotFoundError as e:
        print(f"Error: File tidak ditemukan - {e}")
        print("Pastikan sudah menjalankan normalisasi terlebih dahulu!")
        return None, None

def find_optimal_dbscan_params(data, eps_range=None, min_samples_range=None):
    """
    Mencari parameter optimal untuk DBSCAN
    """
    print(f"\n=== MENCARI PARAMETER OPTIMAL UNTUK DBSCAN ===")
    
    if eps_range is None:
        # Hitung jarak rata-rata untuk menentukan eps range
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors_fit = neighbors.fit(data)
        distances, indices = neighbors_fit.kneighbors(data)
        distances = np.sort(distances[:, 4], axis=0)  # 4th nearest neighbor
        
        # Eps range berdasarkan knee point detection
        eps_range = np.linspace(distances.mean() * 0.1, distances.mean() * 3, 20)
    
    if min_samples_range is None:
        min_samples_range = range(3, 11)
    
    best_params = {}
    best_score = -1
    results = []
    
    print("Testing DBSCAN parameters...")
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(data)
            
            # Skip jika hanya 1 cluster atau semua noise
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            if n_clusters < 4:
                continue
            
            # Hitung silhouette score (exclude noise points)
            if len(set(cluster_labels)) > 1:
                # Filter out noise points for silhouette calculation
                mask = cluster_labels != -1
                if mask.sum() > 1:
                    sil_score = silhouette_score(data[mask], cluster_labels[mask])
                    noise_ratio = (cluster_labels == -1).sum() / len(cluster_labels)
    
                    # Terapkan penalti besar jika noise tinggi
                    penalty_factor = 1 - noise_ratio**2  # kuadrat supaya penalti makin besar
                    penalized_score = sil_score * penalty_factor
                    
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': (cluster_labels == -1).sum(),
                        'noise_ratio': noise_ratio,
                        'silhouette_score': sil_score,
                        'penalized_score': penalized_score
                    })
                    
                    if penalized_score > best_score:
                        best_score = penalized_score
                        best_params = {
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'silhouette_score': sil_score,
                            'penalized_score': penalized_score,
                            'noise_ratio': noise_ratio
                        }
    
    if results:
        results_df = pd.DataFrame(results)
        print(f"\nBest DBSCAN parameters:")
        print(f"eps: {best_params['eps']:.4f}")
        print(f"min_samples: {best_params['min_samples']}")
        print(f"n_clusters: {best_params['n_clusters']}")
        print(f"silhouette_score: {best_params['silhouette_score']:.3f}")
        
        return best_params, results_df
    else:
        print("Tidak ditemukan parameter DBSCAN yang menghasilkan cluster yang baik")
        # Return default parameters
        return {'eps': 0.5, 'min_samples': 5}, pd.DataFrame()

def perform_kmeans_clustering(data, n_clusters=5):
    """
    Melakukan clustering dengan K-Means
    """
    print(f"\n=== K-MEANS CLUSTERING (k={n_clusters}) ===")
    
    start_time = time.time()
    
    # Inisialisasi dan fit model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution time: {execution_time:.4f} seconds")
    
    # Distribusi cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Cluster distribution: {dict(zip(unique, counts))}")
    
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        x_label, y_label = "PCA 1", "PCA 2"
    else:
        data_2d = data.values
        x_label, y_label = data.columns[0], data.columns[1]
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        data_2d[:, 0], data_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, edgecolor='k'
    )
    centers_2d = kmeans.cluster_centers_
    if data.shape[1] > 2:
        centers_2d = pca.transform(centers_2d)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=200, label='Centers')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'K-Means Clustering (k={n_clusters})')
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    results = {
        'model': kmeans,
        'labels': cluster_labels,
        'execution_time': execution_time,
        'n_clusters': n_clusters,
        'cluster_centers': kmeans.cluster_centers_,
        'algorithm': 'kmeans'
    }
    
    return results

def perform_dbscan_clustering(data, eps=0.5, min_samples=5):
    """
    Melakukan clustering dengan DBSCAN
    """
    print(f"\n=== DBSCAN CLUSTERING (eps={eps:.4f}, min_samples={min_samples}) ===")
    
    start_time = time.time()
    
    # Inisialisasi dan fit model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(data)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Analisis hasil
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    # Distribusi cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Cluster distribution: {dict(zip(unique, counts))}")

    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        x_label, y_label = "PCA 1", "PCA 2"
    else:
        data_2d = data.values
        x_label, y_label = data.columns[0], data.columns[1]
    plt.figure(figsize=(8, 6))
    # Noise (-1) akan berwarna hitam
    palette = plt.cm.tab10
    colors = [palette(lbl % 10) if lbl != -1 else 'black' for lbl in cluster_labels]
    plt.scatter(
        data_2d[:, 0], data_2d[:, 1], c=colors, alpha=0.7, edgecolor='k'
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'DBSCAN Clustering (eps={eps:.2f}, min_samples={min_samples})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    results = {
        'model': dbscan,
        'labels': cluster_labels,
        'execution_time': execution_time,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'eps': eps,
        'min_samples': min_samples,
        'algorithm': 'dbscan'
    }
    
    return results

def create_cluster_profiles(data, results_dict, feature_names):
    """
    Membuat profil karakteristik setiap cluster
    """
    print(f"\n=== PROFIL CLUSTER ===")
    
    profiles = {}
    
    for method_name, results in results_dict.items():
        print(f"\n--- {method_name.upper()} CLUSTER PROFILES ---")
        
        data_with_labels = data.copy()
        data_with_labels['cluster'] = results['labels']
        
        # Group by cluster dan hitung statistik
        cluster_stats = data_with_labels.groupby('cluster').agg({
            col: ['mean', 'std', 'count'] for col in feature_names
        }).round(3)
        
        profiles[method_name] = cluster_stats
        
        print(f"Cluster Statistics:")
        print(cluster_stats)
        
        # Hitung persentase untuk setiap cluster
        cluster_counts = data_with_labels['cluster'].value_counts().sort_index()
        cluster_percentages = (cluster_counts / len(data_with_labels) * 100).round(2)
        
        print(f"\nCluster Distribution:")
        for cluster_id, percentage in cluster_percentages.items():
            count = cluster_counts[cluster_id]
            if cluster_id == -1:
                print(f"Noise: {count} samples ({percentage}%)")
            else:
                print(f"Cluster {cluster_id}: {count} samples ({percentage}%)")
        
        means = cluster_stats.xs('mean', axis=1, level=1)
        means.plot(kind='bar', figsize=(12, 6))
        plt.title(f'Rata-rata Fitur per Cluster ({method_name})')
        plt.ylabel('Mean Value')
        plt.xlabel('Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Pie chart distribusi cluster
        plt.figure(figsize=(6, 6))
        labels_pie = [f'Cluster {int(c)}' if c != -1 else 'Noise' for c in cluster_counts.index]
        plt.pie(cluster_percentages, labels=labels_pie, autopct='%1.1f%%', startangle=140)
        plt.title(f'Distribusi Persentase Cluster ({method_name})')
        plt.tight_layout()
        plt.show()
    
    return profiles

def save_clustering_results(kmeans_results, dbscan_results, normalized_datasets, 
                           profiles, output_dir='data/clustering_results'):
    """
    Menyimpan hasil clustering
    """
    # Buat direktori jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Simpan labels untuk setiap metode normalisasi dan algoritma
    for norm_method in normalized_datasets.keys():
        # K-Means labels
        kmeans_labels_df = pd.DataFrame({
            'kmeans_cluster': kmeans_results[norm_method]['labels']
        })
        kmeans_labels_df.to_csv(f'{output_dir}/kmeans_labels_{norm_method}.csv', index=False)
        
        # DBSCAN labels
        dbscan_labels_df = pd.DataFrame({
            'dbscan_cluster': dbscan_results[norm_method]['labels']
        })
        dbscan_labels_df.to_csv(f'{output_dir}/dbscan_labels_{norm_method}.csv', index=False)
    
    # Simpan cluster profiles
    for method_name, profile in profiles.items():
        profile.to_csv(f'{output_dir}/cluster_profile_{method_name}.csv')
    
    print(f"\n=== HASIL CLUSTERING DISIMPAN DI FOLDER {output_dir} ===")

def main():
    """
    Fungsi utama untuk menjalankan clustering
    """
    print("=== CLUSTERING SMARTPHONE USAGE DATA ===\n")
    
    # 1. Load data ternormalisasi
    normalized_datasets, labels = load_normalized_data()
    if normalized_datasets is None:
        return
    
    # Buat direktori hasil
    os.makedirs('data/clustering_results', exist_ok=True)
    
    # Inisialisasi hasil
    kmeans_results = {}
    dbscan_results = {}
    all_profiles = {}
    
    # 2. Lakukan clustering untuk setiap metode normalisasi
    for norm_method, data in normalized_datasets.items():
        print(f"\n{'='*60}")
        print(f"CLUSTERING DENGAN DATA {norm_method.upper()}")
        print(f"{'='*60}")
        
        feature_names = data.columns.tolist()
        k_fixed = 5
        
        # 2a. Cari parameter optimal untuk DBSCAN
        dbscan_params, dbscan_param_results = find_optimal_dbscan_params(data)
        
        # 2b. Lakukan K-Means clustering
        kmeans_result = perform_kmeans_clustering(data, n_clusters=k_fixed)
        kmeans_results[norm_method] = kmeans_result
        
        # 2c. Lakukan DBSCAN clustering
        dbscan_result = perform_dbscan_clustering(
            data, 
            eps=dbscan_params['eps'], 
            min_samples=dbscan_params['min_samples']
        )
        dbscan_results[norm_method] = dbscan_result
        
        # 2d. Buat profil cluster
        results_dict = {
            f'{norm_method}_kmeans': kmeans_result,
            f'{norm_method}_dbscan': dbscan_result
        }
        profiles = create_cluster_profiles(data, results_dict, feature_names)
        all_profiles.update(profiles)
    
    # 3. Simpan semua hasil
    save_clustering_results(kmeans_results, dbscan_results, normalized_datasets, all_profiles)
    
    print(f"\n=== CLUSTERING SELESAI ===")
    print(f"Hasil clustering siap untuk evaluasi!")
    print(f"File hasil tersimpan di folder: data/clustering_results/")
    
    # Return results untuk digunakan oleh evaluation.py
    return kmeans_results, dbscan_results, normalized_datasets

if __name__ == "__main__":
    main()