import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def compare_before_after():
    """Compare results before and after improvements"""
    
    print("="*60)
    print("ANALISIS PERBANDINGAN: SEBELUM VS SESUDAH PERBAIKAN")
    print("="*60)
    
    # Results before improvement (from your original output)
    before_results = {
        'RandomForest': 0.4595,
        'XGBoost': 0.4810,
        'KNN': 0.4952,
        'NaiveBayes': 0.4167
    }
    
    # Load results after improvement
    try:
        after_results_raw = joblib.load('models/all_model_results.pkl')
        after_results = {}
        for name, result in after_results_raw.items():
            after_results[name] = result['accuracy']
        
        # Add ensemble if available
        try:
            best_model_info = joblib.load('models/best_model_info.pkl')
            if best_model_info['name'] == 'Ensemble':
                after_results['Ensemble'] = best_model_info['accuracy']
        except:
            pass
            
    except FileNotFoundError:
        print("❌ Belum ada hasil setelah perbaikan. Jalankan train_model_improved.py terlebih dahulu!")
        return
    
    # Create comparison dataframe
    comparison_data = []
    
    for model in before_results.keys():
        if model in after_results:
            improvement = after_results[model] - before_results[model]
            improvement_pct = (improvement / before_results[model]) * 100
            
            comparison_data.append({
                'Model': model,
                'Before': before_results[model],
                'After': after_results[model],
                'Improvement': improvement,
                'Improvement_Pct': improvement_pct
            })
    
    # Add ensemble if exists
    if 'Ensemble' in after_results:
        comparison_data.append({
            'Model': 'Ensemble',
            'Before': 'N/A',
            'After': after_results['Ensemble'],
            'Improvement': 'New Model',
            'Improvement_Pct': 'N/A'
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display comparison table
    print("\n📊 TABEL PERBANDINGAN AKURASI:")
    print("-" * 70)
    print(f"{'Model':<15} {'Sebelum':<10} {'Sesudah':<10} {'Peningkatan':<12} {'%':<8}")
    print("-" * 70)
    
    for _, row in df_comparison.iterrows():
        model = row['Model']
        before = f"{row['Before']:.4f}" if isinstance(row['Before'], float) else str(row['Before'])
        after = f"{row['After']:.4f}"
        improvement = f"{row['Improvement']:+.4f}" if isinstance(row['Improvement'], float) else str(row['Improvement'])
        improvement_pct = f"{row['Improvement_Pct']:+.2f}%" if isinstance(row['Improvement_Pct'], float) else str(row['Improvement_Pct'])
        
        print(f"{model:<15} {before:<10} {after:<10} {improvement:<12} {improvement_pct:<8}")
    
    print("-" * 70)
    
    # Statistical analysis
    print("\n📈 ANALISIS STATISTIK:")
    print("-" * 40)
    
    numerical_improvements = [row['Improvement'] for row in comparison_data 
                            if isinstance(row['Improvement'], float)]
    
    if numerical_improvements:
        avg_improvement = np.mean(numerical_improvements)
        max_improvement = max(numerical_improvements)
        min_improvement = min(numerical_improvements)
        
        print(f"Rata-rata peningkatan: {avg_improvement:+.4f} ({(avg_improvement/np.mean(list(before_results.values())))*100:+.2f}%)")
        print(f"Peningkatan tertinggi: {max_improvement:+.4f}")
        print(f"Peningkatan terendah: {min_improvement:+.4f}")
        
        # Find best models
        best_before = max(before_results.items(), key=lambda x: x[1])
        best_after = max(after_results.items(), key=lambda x: x[1])
        
        print(f"\n🏆 MODEL TERBAIK:")
        print(f"Sebelum: {best_before[0]} ({best_before[1]:.4f})")
        print(f"Sesudah: {best_after[0]} ({best_after[1]:.4f})")
        print(f"Peningkatan overall: {best_after[1] - best_before[1]:+.4f}")
    
    # Feature analysis
    print("\n🔍 ANALISIS FITUR:")
    print("-" * 40)
    
    try:
        feature_info = joblib.load('models/feature_info.pkl')
        selected_features = feature_info['selected_features']
        all_features = feature_info['all_features']
        
        print(f"Total fitur awal: {len(all_features)}")
        print(f"Fitur terpilih: {len(selected_features)}")
        print(f"Reduksi fitur: {len(all_features) - len(selected_features)} ({((len(all_features) - len(selected_features))/len(all_features)*100):.1f}%)")
        
        print(f"\nFitur terpilih:")
        for i, feature in enumerate(selected_features, 1):
            print(f"  {i:2d}. {feature}")
            
    except FileNotFoundError:
        print("Feature info tidak tersedia")
    
    # Improvement summary
    print("\n" + "="*60)
    print("📋 RINGKASAN PERBAIKAN YANG DILAKUKAN:")
    print("="*60)
    
    improvements_made = [
        "✅ Feature Engineering:",
        "   - Membuat fitur 'Usage_Intensity' gabungan",
        "   - Kategorisasi Screen Time",
        "   - Fitur turunan (Apps_per_Hour, Battery_per_Hour, dll)",
        "   - Kategorisasi Age Group",
        "",
        "✅ Cardinality Reduction:",
        "   - Device Model dikurangi menjadi top 8 + 'Other'",
        "",
        "✅ Feature Selection:",
        "   - SelectKBest dengan f_classif",
        "   - Mengurangi dimensi fitur secara otomatis",
        "",
        "✅ Hyperparameter Tuning:",
        "   - RandomizedSearchCV untuk RF dan XGBoost",
        "   - GridSearchCV untuk KNN",
        "   - Optimasi parameter secara sistematis",
        "",
        "✅ Konsistensi Scaling:",
        "   - Data scaled untuk KNN dan Naive Bayes",
        "   - Data unscaled untuk tree-based models",
        "",
        "✅ Ensemble Method:",
        "   - Voting Classifier dari model terbaik"
    ]
    
    for improvement in improvements_made:
        print(improvement)
    
    # Recommendations
    print("\n" + "="*60)
    print("💡 REKOMENDASI LANJUTAN:")
    print("="*60)
    
    recommendations = [
        "1. 🎯 Cross-Validation yang lebih robust:",
        "   - Gunakan StratifiedKFold dengan k=5 atau k=10",
        "   - Validasi konsistensi performa across folds",
        "",
        "2. 🔄 Advanced Feature Engineering:",
        "   - Polynomial features untuk interaksi non-linear",
        "   - Time-based features jika ada data temporal",
        "   - Domain-specific features berdasarkan pengetahuan smartphone usage",
        "",
        "3. 🤖 Advanced Models:",
        "   - Neural Networks (MLPClassifier)",
        "   - Gradient Boosting variants (LightGBM, CatBoost)",
        "   - Stacking ensemble (lebih sophisticated dari voting)",
        "",
        "4. 📊 Data Analysis:",
        "   - Analisis korelasi fitur terhadap target",
        "   - Identifikasi outliers dan treatment",
        "   - Feature importance analysis",
        "",
        "5. ⚖️ Class Balancing:",
        "   - SMOTE untuk oversampling minority classes",
        "   - Class weight tuning yang lebih optimal",
        "",
        "6. 🎛️ Advanced Hyperparameter Tuning:",
        "   - Bayesian Optimization (Optuna, Hyperopt)",
        "   - Automated ML (AutoML) frameworks"
    ]
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "="*60)
    
    return df_comparison

def plot_comparison():
    """Create visualization of before/after comparison"""
    try:
        # Load data for plotting
        before_results = {
            'RandomForest': 0.4595,
            'XGBoost': 0.4810,
            'KNN': 0.4952,
            'NaiveBayes': 0.4167
        }
        
        after_results_raw = joblib.load('models/all_model_results.pkl')
        after_results = {}
        for name, result in after_results_raw.items():
            after_results[name] = result['accuracy']
        
        # Create comparison plot
        models = list(before_results.keys())
        before_scores = [before_results[model] for model in models]
        after_scores = [after_results.get(model, 0) for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width/2, before_scores, width, label='Sebelum Perbaikan', 
                      color='lightcoral', alpha=0.8)
        bars2 = ax.bar(x + width/2, after_scores, width, label='Sesudah Perbaikan', 
                      color='lightblue', alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Akurasi')
        ax.set_title('Perbandingan Akurasi Model: Sebelum vs Sesudah Perbaikan')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("📊 Grafik perbandingan disimpan sebagai 'model_comparison.png'")
        plt.show()
        
    except Exception as e:
        print(f"Error creating plot: {e}")

def analyze_feature_importance():
    """Analyze feature importance from best models"""
    try:
        print("\n" + "="*50)
        print("🔍 ANALISIS FEATURE IMPORTANCE")
        print("="*50)
        
        # Load models and feature info
        all_results = joblib.load('models/all_model_results.pkl')
        feature_info = joblib.load('models/feature_info.pkl')
        selected_features = feature_info['selected_features']
        
        # Analyze Random Forest feature importance
        if 'RandomForest' in all_results:
            rf_model = all_results['RandomForest']['model']
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': selected_features,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print("\n🌲 Random Forest - Top 10 Feature Importance:")
                print("-" * 50)
                for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
                    print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
        
        # Analyze XGBoost feature importance
        if 'XGBoost' in all_results:
            xgb_model = all_results['XGBoost']['model']
            if hasattr(xgb_model, 'feature_importances_'):
                importances = xgb_model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': selected_features,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print("\n🚀 XGBoost - Top 10 Feature Importance:")
                print("-" * 50)
                for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
                    print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
                    
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")

if __name__ == "__main__":
    # Run comparison analysis
    comparison_df = compare_before_after()
    
    # Create visualization
    plot_comparison()
    
    # Analyze feature importance
    analyze_feature_importance()
    
    print("\n✅ Analisis perbandingan selesai!")
    print("📁 File yang dihasilkan:")
    print("   - model_comparison.png (grafik perbandingan)")
    print("   - All model results tersimpan di models/")