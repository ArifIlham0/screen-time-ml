"""
Script untuk menjalankan seluruh pipeline machine learning secara otomatis
Menggabungkan semua perbaikan yang diusulkan
"""

import os
import sys
import time
from datetime import datetime

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def run_preprocessing():
    """Run improved preprocessing"""
    print("\n" + "="*60)
    print("🔄 STEP 1: PREPROCESSING DATA")
    print("="*60)
    
    try:
        # Import and run preprocessing
        from preprocessing_improved import preprocess_data
        preprocess_data()
        print("✅ Preprocessing completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return False

def run_training():
    """Run improved model training"""
    print("\n" + "="*60)
    print("🤖 STEP 2: TRAINING MODELS WITH HYPERPARAMETER TUNING")
    print("="*60)
    
    try:
        # Import and run training
        from test.train_models import main
        main()
        print("✅ Model training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error in training: {e}")
        return False

def run_comparison():
    """Run comparison analysis"""
    print("\n" + "="*60)
    print("📊 STEP 3: COMPARISON ANALYSIS")
    print("="*60)
    
    try:
        # Import and run comparison
        from comparison_analysis import compare_before_after, plot_comparison, analyze_feature_importance
        
        compare_before_after()
        plot_comparison()
        analyze_feature_importance()
        
        print("✅ Comparison analysis completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        return False

def run_prediction_test():
    """Test prediction with sample data"""
    print("\n" + "="*60)
    print("🎯 STEP 4: TESTING PREDICTION")
    print("="*60)
    
    try:
        import joblib
        import numpy as np
        from test.predict_usage import create_user_features, preprocess_user_input
        
        # Load model
        model = joblib.load('models/best_model.pkl')
        model_info = joblib.load('models/best_model_info.pkl')
        
        # Test with sample data
        sample_data = {
            'app_usage': 300,
            'screen_time': 6.5,
            'battery_drain': 2500,
            'num_apps': 15,
            'data_usage': 800,
            'age': 25,
            'device_model': 'iPhone',
            'os_type': 'iOS',
            'gender': 'Male'
        }
        
        print("Testing prediction with sample data:")
        for key, value in sample_data.items():
            print(f"  {key}: {value}")
        
        # Create features and predict
        df = create_user_features(**sample_data)
        X_scaled, X_selected = preprocess_user_input(df)
        
        try:
            prediction = model.predict(X_scaled)[0]
        except:
            prediction = model.predict(X_selected)[0]
        
        label_map = {
            0: "Penggunaan Sangat Rendah",
            1: "Penggunaan Rendah", 
            2: "Penggunaan Sedang",
            3: "Penggunaan Tinggi",
            4: "Penggunaan Sangat Tinggi"
        }
        
        print(f"\n🎯 Prediction Result: {label_map[prediction]}")
        print(f"📊 Model Used: {model_info['name']}")
        print(f"🎯 Model Accuracy: {model_info['accuracy']:.4f}")
        
        print("✅ Prediction test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in prediction test: {e}")
        return False

def print_summary():
    """Print final summary"""
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\n📋 FILES CREATED:")
    print("-" * 40)
    
    files_to_check = [
        ('data/preprocessed_data_scaled.pkl', 'Scaled preprocessed data'),
        ('data/preprocessed_data_unscaled.pkl', 'Unscaled preprocessed data'),
        ('models/best_model.pkl', 'Best trained model'),
        ('models/best_model_info.pkl', 'Best model information'),
        ('models/scaler.pkl', 'Data scaler'),
        ('models/label_encoder.pkl', 'Label encoder'),
        ('models/feature_selector.pkl', 'Feature selector'),
        ('models/feature_info.pkl', 'Feature information'),
        ('models/all_model_results.pkl', 'All model results'),
        ('model_comparison.png', 'Comparison visualization')
    ]
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path:<35} - {description}")
        else:
            print(f"❌ {file_path:<35} - {description} (Missing)")
    
    print("\n🚀 NEXT STEPS:")
    print("-" * 40)
    print("1. 📊 Review the comparison analysis results")
    print("2. 📈 Check model_comparison.png for visual comparison")
    print("3. 🎯 Test prediction with predict_user_input_improved.py")
    print("4. 📝 Document the improvements in your thesis")
    print("5. 🔬 Consider implementing advanced recommendations")
    
    print("\n💡 THESIS WRITING TIPS:")
    print("-" * 40)
    print("• Document the feature engineering process")
    print("• Explain hyperparameter tuning methodology")
    print("• Include before/after comparison tables")
    print("• Discuss feature selection impact")
    print("• Analyze feature importance results")
    print("• Present ensemble method benefits")
    
    print("\n" + "="*80)

def main():
    """Main pipeline execution"""
    start_time = time.time()
    
    print("🚀 STARTING COMPLETE MACHINE LEARNING PIPELINE")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check if data file exists
    if not os.path.exists('data/user_behavior_dataset.csv'):
        print("❌ ERROR: data/user_behavior_dataset.csv not found!")
        print("Please ensure your dataset is in the correct location.")
        return False
    
    # Create necessary directories
    create_directories()
    
    # Step 1: Preprocessing
    if not run_preprocessing():
        print("❌ Pipeline stopped due to preprocessing error")
        return False
    
    # Step 2: Training
    if not run_training():
        print("❌ Pipeline stopped due to training error")
        return False
    
    # Step 3: Comparison
    if not run_comparison():
        print("⚠️ Comparison analysis failed, but continuing...")
    
    # Step 4: Prediction test
    if not run_prediction_test():
        print("⚠️ Prediction test failed, but pipeline mostly completed...")
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n⏱️ Total execution time: {execution_time:.2f} seconds")
    
    # Print summary
    print_summary()
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎊 PIPELINE COMPLETED SUCCESSFULLY!")
        print("Check the results and run predict_user_input_improved.py for interactive prediction!")
    else:
        print("\n💥 PIPELINE FAILED!")
        print("Check the error messages above and fix the issues.")