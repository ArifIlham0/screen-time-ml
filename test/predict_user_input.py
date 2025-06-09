import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class UsageIntensityPredictor:
    def __init__(self):
        # Load the trained model and preprocessing objects
        self.model = joblib.load('models/best_usage_cluster_model.pkl')
        self.scaler = joblib.load('models/cluster_scaler.pkl')
        self.features = ['Screen On Time (hours/day)', 
                        'App Usage Time (min/day)', 
                        'Open Apps']
        
    def preprocess_input(self, screen_time, app_usage, open_apps):
        """Preprocess user input to match training data format"""
        # Convert app usage from hours to minutes (as in training data)
        app_usage_min = app_usage * 60
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[screen_time, app_usage_min, open_apps]],
                                 columns=self.features)
        
        # Create additional features (same as in preprocessing)
        input_data['Apps_per_Hour'] = input_data['Open Apps'] / (input_data['Screen On Time (hours/day)'] + 0.1)
        input_data['Usage_per_App'] = input_data['App Usage Time (min/day)'] / (input_data['Open Apps'] + 1)
        
        # Select the same features used in training
        X = input_data[self.features + ['Apps_per_Hour', 'Usage_per_App']]
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict_intensity(self, screen_time, app_usage, open_apps):
        """Predict usage intensity level"""
        # Preprocess the input
        X = self.preprocess_input(screen_time, app_usage, open_apps)
        
        # Make prediction
        prediction = self.model.predict(X)
        
        # Map numeric prediction to label
        intensity_levels = {
            0: 'Very Low',
            1: 'Low',
            2: 'Medium', 
            3: 'High',
            4: 'Very High'
        }
        
        return intensity_levels[prediction[0]]
    
    def predict_proba(self, screen_time, app_usage, open_apps):
        """Get probability estimates for each class"""
        X = self.preprocess_input(screen_time, app_usage, open_apps)
        proba = self.model.predict_proba(X)[0]
        
        classes = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        return dict(zip(classes, proba))

def main():
    # Initialize predictor
    predictor = UsageIntensityPredictor()
    
    print("Smartphone Usage Intensity Predictor")
    print("-----------------------------------")
    
    # Get user input
    screen_time = float(input("Enter screen time (hours/day): "))
    app_usage = float(input("Enter app usage (hours/day): "))
    open_apps = int(input("Enter number of apps opened per day: "))
    
    # Make prediction
    intensity = predictor.predict_intensity(screen_time, app_usage, open_apps)
    probabilities = predictor.predict_proba(screen_time, app_usage, open_apps)
    
    # Display results
    print("\nPrediction Results:")
    print(f"Usage Intensity: {intensity}")
    
    print("\nProbability Estimates:")
    for level, prob in probabilities.items():
        print(f"{level:10s}: {prob:.2%}")

if __name__ == "__main__":
    main()