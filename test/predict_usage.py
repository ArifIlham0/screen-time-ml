import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class SmartphoneUsagePredictor:
    def __init__(self):
        """Initialize the predictor by loading the trained model and required components"""
        try:
            # Load the best trained model
            self.model_info = joblib.load('models/best_model.pkl')
            self.model = self.model_info['model']
            self.scaler = self.model_info['scaler']
            self.label_encoder = self.model_info['label_encoder']
            self.selected_features = self.model_info['selected_features']
            
            # Load additional required components
            self.feature_stats = joblib.load('models/feature_stats.pkl')
            
            # Load class labels mapping
            self.intensity_labels = {
                0: 'Very Low',
                1: 'Low', 
                2: 'Medium',
                3: 'High',
                4: 'Very High'
            }
            
            print(f"Model loaded successfully!")
            print(f"Target: {self.model_info['target_name']}")
            print(f"Model Type: {self.model_info['model_name']}")
            print(f"Model Accuracy: {self.model_info['accuracy']:.4f}")
            
        except FileNotFoundError as e:
            print(f"Error loading model files: {e}")
            print("Please run train_models.py first to train the models.")
            raise
    
    def create_features_from_input(self, app_usage_time, screen_time, open_apps, age):
        """
        Create all required features from the 4 user inputs
        """
        # Create a dictionary to store all features
        features = {}
        
        # Basic input features
        features['Age'] = age
        features['App Usage Time (min/day)'] = app_usage_time
        features['Screen On Time (hours/day)'] = screen_time
        features['Open Apps'] = open_apps
        
        # For missing features, we'll use reasonable defaults or statistical estimates
        # Battery Drain estimation (based on typical smartphone usage patterns)
        # Typically 200-400 mAh per hour of screen time
        estimated_battery_drain = screen_time * 300 + app_usage_time * 2
        features['Battery Drain (mAh/day)'] = estimated_battery_drain
        
        # Data Usage estimation (based on typical patterns)
        # Typically 50-200 MB per hour of screen time
        estimated_data_usage = screen_time * 100 + open_apps * 20
        features['Data Usage (MB/day)'] = estimated_data_usage
        
        # Now create derived features (same as in preprocessing)
        features['Apps_per_Hour'] = open_apps / (screen_time + 0.1)
        features['Battery_per_Hour'] = estimated_battery_drain / (screen_time + 0.1)
        features['Usage_per_App'] = app_usage_time / (open_apps + 1)
        features['Data_per_Hour'] = estimated_data_usage / (screen_time + 0.1)
        
        # Usage Intensity (using the same formula as preprocessing)
        # Need to normalize using the max values from training data
        app_usage_norm = app_usage_time / self.feature_stats['App Usage Time (min/day)_max']
        battery_norm = estimated_battery_drain / self.feature_stats['Battery Drain (mAh/day)_max']
        screen_norm = screen_time / self.feature_stats['Screen On Time (hours/day)_max']
        data_norm = estimated_data_usage / self.feature_stats['Data Usage (MB/day)_max']
        
        features['Usage_Intensity'] = (
            app_usage_norm * 0.3 +
            battery_norm * 0.25 +
            screen_norm * 0.25 +
            data_norm * 0.2
        )
        
        # Efficiency features
        features['Battery_Efficiency'] = app_usage_time / (estimated_battery_drain + 1)
        features['Data_Efficiency'] = app_usage_time / (estimated_data_usage + 1)
        
        # Session duration estimation
        features['Avg_Session_Duration'] = (screen_time * 60) / (open_apps + 1)
        
        # Variance features (using statistical approximation)
        features['Usage_Variance'] = abs(app_usage_time - self.feature_stats['App Usage Time (min/day)_mean'])
        features['Battery_Variance'] = abs(estimated_battery_drain - self.feature_stats['Battery Drain (mAh/day)_mean'])
        
        # Power User indicator
        app_usage_75th = self.feature_stats['App Usage Time (min/day)_max'] * 0.75  # Approximation
        battery_75th = self.feature_stats['Battery Drain (mAh/day)_max'] * 0.75    # Approximation
        
        features['Is_Power_User'] = int(
            (app_usage_time > app_usage_75th) and 
            (estimated_battery_drain > battery_75th)
        )
        
        # Polynomial features
        features['App_Usage_Squared'] = app_usage_time ** 2
        features['Battery_Usage_Squared'] = estimated_battery_drain ** 2
        features['Screen_Time_Squared'] = screen_time ** 2
        
        # Interaction features
        features['Age_Screen_Interaction'] = age * screen_time
        features['Battery_Data_Interaction'] = estimated_battery_drain * estimated_data_usage
        
        # Age group encoding (same bins as preprocessing)
        if age <= 11:
            age_group = 'Children'
        elif age <= 25:
            age_group = 'Teenager'
        elif age <= 45:  
            age_group = 'Adult'
        else:
            age_group = 'Elderly'
        
        # Age group one-hot encoding
        features['Age_Group_Adult'] = 1 if age_group == 'Adult' else 0
        features['Age_Group_Teenager'] = 1 if age_group == 'Teenager' else 0
        features['Age_Group_Children'] = 1 if age_group == 'Children' else 0
        features['Age_Group_Elderly'] = 1 if age_group == 'Elderly' else 0

        # Screen time categories
        if screen_time <= 3:
            screen_cat = 'Low'
        elif screen_time <= 6:
            screen_cat = 'Medium'
        elif screen_time <= 9:
            screen_cat = 'High'
        else:
            screen_cat = 'Extreme'
            
        features['Screen_Time_Category_High'] = 1 if screen_cat == 'High' else 0
        features['Screen_Time_Category_Low'] = 1 if screen_cat == 'Low' else 0
        features['Screen_Time_Category_Medium'] = 1 if screen_cat == 'Medium' else 0
        
        # Data usage categories
        if estimated_data_usage <= 500:
            data_cat = 'Light'
        elif estimated_data_usage <= 1500:
            data_cat = 'Moderate'
        elif estimated_data_usage <= 3000:
            data_cat = 'Heavy'
        else:
            data_cat = 'Extreme'
            
        features['Data_Usage_Category_Heavy'] = 1 if data_cat == 'Heavy' else 0
        features['Data_Usage_Category_Light'] = 1 if data_cat == 'Light' else 0
        features['Data_Usage_Category_Moderate'] = 1 if data_cat == 'Moderate' else 0
        
        # Add default values for any other categorical features that might be missing
        # Gender (default to most common or neutral)
        features['Gender_Male'] = 0  # Default assumption
        
        # Device and OS features (using frequency encoding - set to average frequency)
        features['Device Model_freq'] = 0.1  # Average frequency
        features['Operating System_freq'] = 0.5  # Assume common OS
        features['Device_OS_Combo_freq'] = 0.05  # Average combo frequency
        
        return features
    
    def predict_usage_intensity(self, app_usage_time, screen_time, open_apps, age):
        """
        Predict smartphone usage intensity from user inputs
        
        Parameters:
        - app_usage_time: App Usage Time in minutes per day
        - screen_time: Screen On Time in hours per day  
        - open_apps: Number of apps opened
        - age: User age
        
        Returns:
        - prediction: Usage intensity category (Very Low, Low, Medium, High, Very High)
        - confidence: Prediction confidence score
        """
        
        # Input validation
        if app_usage_time < 0 or screen_time < 0 or open_apps < 0 or age < 0:
            raise ValueError("All input values must be non-negative")
        
        if age > 100:
            raise ValueError("Age seems unrealistic (>100)")
            
        if screen_time > 24:
            raise ValueError("Screen time cannot exceed 24 hours per day")
            
        if app_usage_time > screen_time * 60:
            print("Warning: App usage time is greater than total screen time")
        
        # Create all features from inputs
        all_features = self.create_features_from_input(app_usage_time, screen_time, open_apps, age)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([all_features])
        
        # Select only the features that were used in training
        # Fill missing features with 0 (they will be handled by scaling)
        feature_vector = []
        for feature_name in self.selected_features:
            if feature_name in feature_df.columns:
                feature_vector.append(feature_df[feature_name].iloc[0])
            else:
                feature_vector.append(0)  # Default value for missing features
        
        # Convert to numpy array and reshape
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_array)
        
        # Make prediction
        prediction_encoded = self.model.predict(feature_scaled)[0]
        
        # Get prediction probabilities for confidence
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(feature_scaled)[0]
            print(f"Prediction probabilities 1: {probabilities}")
            print(f"Prediction probabilities 2: {self.model.predict_proba}")
            sorted_probs = np.sort(probabilities)[::-1]
            # Confidence = jumlah dua probabilitas tertinggi, maksimal 1.0
            confidence = min(sorted_probs[0] + sorted_probs[1], 1.0)
        else:
            confidence = 0.8  # Default confidence for models without predict_proba
        
        # Decode prediction
        if self.label_encoder:
            # If we have a label encoder, use it
            try:
                prediction_label = self.label_encoder.inverse_transform([prediction_encoded])[0]
            except:
                # Fallback to intensity labels
                prediction_label = self.intensity_labels.get(prediction_encoded, 'Unknown')
        else:
            # Use intensity labels mapping
            prediction_label = self.intensity_labels.get(prediction_encoded, 'Unknown')
        
        return prediction_label, confidence
    
    def get_usage_insights(self, app_usage_time, screen_time, open_apps, age):
        """
        Provide additional insights about the user's smartphone usage
        """
        insights = []
        
        # Calculate some metrics
        apps_per_hour = open_apps / (screen_time + 0.1)
        usage_per_app = app_usage_time / (open_apps + 1)
        
        # Age-based insights
        if age < 25:
            insights.append("Young users typically have higher smartphone engagement")
        elif age > 45:
            insights.append("Mature users tend to have more focused app usage patterns")
        
        # Screen time insights
        if screen_time > 8:
            insights.append("Very high screen time - consider digital wellness practices")
        elif screen_time < 2:
            insights.append("Low screen time - indicates minimal smartphone usage")
        
        # App usage insights
        if apps_per_hour > 10:
            insights.append("High app switching behavior - might indicate multitasking or restlessness")
        elif usage_per_app > 30:
            insights.append("Deep engagement with individual apps")
        
        # Usage efficiency
        if app_usage_time > screen_time * 60:
            insights.append("App usage time exceeds screen time - please verify input data")
        
        return insights

def main():
    """Main function to demonstrate usage"""
    try:
        # Initialize predictor
        predictor = SmartphoneUsagePredictor()
        
        print(f"\n{'='*60}")
        print(f"SMARTPHONE USAGE INTENSITY PREDICTOR")
        print(f"{'='*60}")
        
        # Interactive mode
        while True:
            try:
                print(f"\nEnter smartphone usage details:")
                
                app_usage = float(input("App Usage Time (minutes/day): "))
                screen_time = float(input("Screen On Time (hours/day): "))
                open_apps = int(input("Number of Open Apps: "))
                age = int(input("Age: "))
                
                # Make prediction
                prediction, confidence = predictor.predict_usage_intensity(
                    app_usage, screen_time, open_apps, age
                )
                
                # Get insights
                insights = predictor.get_usage_insights(app_usage, screen_time, open_apps, age)
                
                # Display results
                print(f"\n{'='*40}")
                print(f"PREDICTION RESULTS")
                print(f"{'='*40}")
                print(f"Usage Intensity: {prediction}")
                print(f"Confidence: {confidence:.2%}")
                
                if insights:
                    print(f"\nInsights:")
                    for insight in insights:
                        print(f"• {insight}")
                
                # Ask for another prediction
                another = input(f"\nMake another prediction? (y/n): ").lower()
                if another != 'y':
                    break
                    
            except ValueError as e:
                print(f"Invalid input: {e}")
            except Exception as e:
                print(f"Error: {e}")
        
        print(f"\nThank you for using the Smartphone Usage Predictor!")
        
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")

if __name__ == "__main__":
    main()