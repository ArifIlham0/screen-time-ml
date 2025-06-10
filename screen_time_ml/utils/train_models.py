import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate_models():
    print("Loading preprocessed data...")
    try:
        processed_data = joblib.load('data/advanced_preprocessed_data.pkl')
    except FileNotFoundError:
        print("Error: Preprocessed data not found. Please run preprocessing_improved.py first.")
        return
    
    best_models = {}
    best_accuracy = 0
    best_model_info = None
    
    # Test different target variables to find the best one
    for target_name, data in processed_data.items():
        print(f"\n{'='*50}")
        print(f"Training models for target: {target_name}")
        print(f"{'='*50}")
        
        X_train = data['X_train_scaled']
        X_test = data['X_test_scaled']
        y_train = data['y_train']
        y_test = data['y_test']
        
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'KNN': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        }
        
        # Hyperparameter grids for tuning
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'Naive Bayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        }
        
        target_models = {}
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            # Hyperparameter tuning
            grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Store model info
            target_models[model_name] = {
                'model': best_model,
                'accuracy': accuracy,
                'cv_accuracy': cv_scores.mean(),
                'predictions': y_pred,
                'best_params': grid_search.best_params_
            }
            
            # Check if this is the best model overall
            if accuracy > best_accuracy and accuracy >= 0.8:
                best_accuracy = accuracy
                best_model_info = {
                    'target_name': target_name,
                    'model_name': model_name,
                    'model': best_model,
                    'accuracy': accuracy,
                    'cv_accuracy': cv_scores.mean(),
                    'data_info': data,
                    'best_params': grid_search.best_params_
                }
        
        best_models[target_name] = target_models
        
        # Print detailed results for best model in this target
        best_target_model = max(target_models.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest model for {target_name}: {best_target_model[0]} (Accuracy: {best_target_model[1]['accuracy']:.4f})")
        
        # Classification report for best model
        if best_target_model[1]['accuracy'] >= 0.8:
            print(f"\nClassification Report for {best_target_model[0]}:")
            print(classification_report(y_test, best_target_model[1]['predictions']))
    
    # # Save the best overall model
    # if best_model_info and best_model_info['accuracy'] >= 0.8:
    #     print(f"\n{'='*60}")
    #     print(f"BEST OVERALL MODEL FOUND!")
    #     print(f"{'='*60}")
    #     print(f"Target: {best_model_info['target_name']}")
    #     print(f"Model: {best_model_info['model_name']}")
    #     print(f"Accuracy: {best_model_info['accuracy']:.4f}")
    #     print(f"CV Accuracy: {best_model_info['cv_accuracy']:.4f}")
    #     print(f"Best Parameters: {best_model_info['best_params']}")
        
    #     # Save the best model and related info
    #     model_info = {
    #         'model': best_model_info['model'],
    #         'target_name': best_model_info['target_name'],
    #         'model_name': best_model_info['model_name'],
    #         'accuracy': best_model_info['accuracy'],
    #         'scaler': best_model_info['data_info']['scaler'],
    #         'label_encoder': best_model_info['data_info']['label_encoder'],
    #         'selected_features': best_model_info['data_info']['selected_features'],
    #         'feature_selector_f': best_model_info['data_info']['feature_selector_f'],
    #         'feature_selector_mi': best_model_info['data_info']['feature_selector_mi'],
    #         'best_params': best_model_info['best_params']
    #     }
        
    #     joblib.dump(model_info, 'models/best_model.pkl')
    #     print(f"\nBest model saved to 'models/best_model.pkl'")
        
    #     # Save feature importance if it's Random Forest
    #     if best_model_info['model_name'] == 'Random Forest':
    #         feature_importance = pd.DataFrame({
    #             'feature': best_model_info['data_info']['selected_features'],
    #             'importance': best_model_info['model'].feature_importances_
    #         }).sort_values('importance', ascending=False)
            
    #         print(f"\nTop 10 Feature Importances:")
    #         print(feature_importance.head(10))
            
    #         # Save feature importance
    #         feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    # else:
    #     print(f"\n{'='*60}")
    #     print(f"NO MODEL ACHIEVED MINIMUM ACCURACY OF 0.8")
    #     print(f"{'='*60}")
    #     if best_model_info:
    #         print(f"Best achieved accuracy: {best_model_info['accuracy']:.4f}")
    #     else:
    #         print("No models were trained successfully.")
        # Save the best model for intensity_target only
    if 'intensity_target' in best_models:
        best_target_model = max(best_models['intensity_target'].items(), key=lambda x: x[1]['accuracy'])
        best_model = best_target_model[1]['model']
        best_model_data = processed_data['intensity_target']
        print(f"\nSaving best model for intensity_target: {best_target_model[0]} (Accuracy: {best_target_model[1]['accuracy']:.4f})")
        model_info = {
            'model': best_model,
            'target_name': 'intensity_target',
            'model_name': best_target_model[0],
            'accuracy': best_target_model[1]['accuracy'],
            'scaler': best_model_data['scaler'],
            'label_encoder': best_model_data['label_encoder'],
            'selected_features': best_model_data['selected_features'],
            'feature_selector_f': best_model_data['feature_selector_f'],
            'feature_selector_mi': best_model_data['feature_selector_mi'],
            'best_params': best_target_model[1]['best_params']
        }
        joblib.dump(model_info, 'models/best_model.pkl')
        print(f"\nBest intensity_target model saved to 'models/best_model.pkl'")

        # Save feature importance if it's Random Forest
        if best_target_model[0] == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': best_model_data['selected_features'],
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\nTop 20 Feature Importances:")
            print(feature_importance.head(20))
            feature_importance.to_csv('models/feature_importance.csv', index=False)
    else:
        print("\nNo intensity_target model found to save.")
    
    # Save all results for analysis
    joblib.dump(best_models, 'models/all_model_results.pkl')
    print(f"\nAll model results saved to 'models/all_model_results.pkl'")
    
    return best_model_info

def analyze_results():
    """Analyze and display comprehensive results"""
    try:
        all_results = joblib.load('models/all_model_results.pkl')
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE RESULTS ANALYSIS")
        print(f"{'='*60}")
        
        for target_name, models in all_results.items():
            print(f"\nTarget: {target_name}")
            print("-" * 40)
            for model_name, results in models.items():
                print(f"{model_name:15}: {results['accuracy']:.4f} (CV: {results['cv_accuracy']:.4f})")
        
        # Find overall best
        # best_overall = None
        best_acc = 0
        
        for target_name, models in all_results.items():
            for model_name, results in models.items():
                if results['accuracy'] > best_acc:
                    best_acc = results['accuracy']
                    best_overall = (target_name, model_name, results['accuracy'])
        
        # if best_overall:
        #     print(f"\nOverall Best: {best_overall[1]} on {best_overall[0]} (Accuracy: {best_overall[2]:.4f})")
            
    except FileNotFoundError:
        print("No results file found. Please run training first.")

if __name__ == "__main__":
    print("Starting model training and evaluation...")
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    # Train models
    best_model = train_and_evaluate_models()
    
    # Analyze results
    analyze_results()