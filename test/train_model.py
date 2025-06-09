import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

def load_advanced_data():
    """Load advanced preprocessed data"""
    return joblib.load('data/advanced_preprocessed_data.pkl')

def train_all_models(X_train_scaled, X_train_unscaled, y_train, 
                    X_test_scaled, X_test_unscaled, y_test):
    """Train all three models and return results"""
    results = {}
    
    # 1. Random Forest
    print("\n=== Training Random Forest ===")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=10, cv=3, 
                                 scoring='accuracy', random_state=42, n_jobs=-1)
    rf_search.fit(X_train_unscaled, y_train)
    
    best_rf = rf_search.best_estimator_
    rf_preds = best_rf.predict(X_test_unscaled)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    results['RandomForest'] = {
        'model': best_rf,
        'accuracy': rf_acc,
        'predictions': rf_preds
    }
    
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # 2. KNN
    print("\n=== Training KNN ===")
    knn_params = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier()
    knn_search = RandomizedSearchCV(knn, knn_params, n_iter=10, cv=3,
                                  scoring='accuracy', random_state=42)
    knn_search.fit(X_train_scaled, y_train)
    
    best_knn = knn_search.best_estimator_
    knn_preds = best_knn.predict(X_test_scaled)
    knn_acc = accuracy_score(y_test, knn_preds)
    
    results['KNN'] = {
        'model': best_knn,
        'accuracy': knn_acc,
        'predictions': knn_preds
    }
    
    print(f"KNN Accuracy: {knn_acc:.4f}")

    # 3. Naive Bayes
    print("\n=== Training Naive Bayes ===")
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    nb_preds = nb.predict(X_test_scaled)
    nb_acc = accuracy_score(y_test, nb_preds)
    
    results['NaiveBayes'] = {
        'model': nb,
        'accuracy': nb_acc,
        'predictions': nb_preds
    }
    
    print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
    
    return results

def check_all_models_above_threshold(results, threshold=0.8):
    """Check if all models have accuracy above threshold"""
    return all(result['accuracy'] > threshold for result in results.values())

def train_models_for_target(target_name, target_data):
    """Train models for a specific target"""
    print(f"\n" + "="*60)
    print(f"TRAINING MODELS FOR TARGET: {target_name.upper()}")
    print("="*60)
    
    # Extract data
    X_train_scaled = target_data['X_train_scaled']
    X_test_scaled = target_data['X_test_scaled']
    X_train_unscaled = target_data['X_train_unscaled']
    X_test_unscaled = target_data['X_test_unscaled']
    y_train = target_data['y_train']
    y_test = target_data['y_test']
    
    print(f"Training set size: {X_train_scaled.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    # Train all models
    results = train_all_models(
        X_train_scaled, X_train_unscaled, y_train,
        X_test_scaled, X_test_unscaled, y_test
    )
    
    # Check if all models meet threshold
    all_above_threshold = check_all_models_above_threshold(results)
    
    return results, y_test, all_above_threshold

def evaluate_and_save_results(all_target_results):
    """Evaluate results and save models"""
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    best_models = {}
    
    for target_name, (results, y_test, _) in all_target_results.items():
        print(f"\n--- {target_name.upper()} ---")
        best_acc = 0
        best_model_name = None
        
        for model_name, result in results.items():
            acc = result['accuracy']
            print(f"{model_name:15s}: Accuracy = {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_model_name = model_name
        
        # Save the best model for this target
        if best_model_name:
            best_model = results[best_model_name]['model']
            model_path = f'models/best_{target_name}_model.pkl'
            joblib.dump(best_model, model_path)
            print(f"Saved best model ({best_model_name}) to {model_path}")
            best_models[target_name] = {
                'model_name': best_model_name,
                'accuracy': best_acc,
                'path': model_path
            }
    
    return best_models

def main():
    """Main training function"""
    print("Loading preprocessed data...")
    processed_data = load_advanced_data()
    
    print(f"Available targets: {list(processed_data.keys())}")
    
    all_results = {}
    stop_training = False
    
    for target_name, target_data in processed_data.items():
        if stop_training:
            break
            
        try:
            print(f"\nProcessing target: {target_name}")
            results, y_test, all_above_threshold = train_models_for_target(target_name, target_data)
            all_results[target_name] = (results, y_test, all_above_threshold)
            
            if all_above_threshold:
                print("\n🎉 ALL MODELS ACHIEVED >0.8 ACCURACY! 🎉")
                stop_training = True
                
        except Exception as e:
            print(f"Error processing {target_name}: {str(e)}")
            continue
    
    # Evaluate and save results
    if all_results:
        best_models = evaluate_and_save_results(all_results)
        
        # Print final summary
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        for target, info in best_models.items():
            print(f"{target:20s}: {info['model_name']} (Accuracy: {info['accuracy']:.4f})")
    else:
        print("No successful model training completed.")

if __name__ == "__main__":
    main()