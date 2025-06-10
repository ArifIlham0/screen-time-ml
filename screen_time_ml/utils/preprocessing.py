import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(df):
    print("Creating advanced features...")
    
    # 1. Basic derived features
    df['Apps_per_Hour'] = df['Open Apps'] / (df['Screen On Time (hours/day)'] + 0.1)
    df['Battery_per_Hour'] = df['Battery Drain (mAh/day)'] / (df['Screen On Time (hours/day)'] + 0.1)
    df['Usage_per_App'] = df['App Usage Time (min/day)'] / (df['Open Apps'] + 1)
    df['Data_per_Hour'] = df['Data Usage (MB/day)'] / (df['Screen On Time (hours/day)'] + 0.1)
    
    # 2. Intensity and Efficiency Features
    df['Usage_Intensity'] = (
        (df['App Usage Time (min/day)'] / df['App Usage Time (min/day)'].max()) * 0.3 +
        (df['Battery Drain (mAh/day)'] / df['Battery Drain (mAh/day)'].max()) * 0.25 +
        (df['Screen On Time (hours/day)'] / df['Screen On Time (hours/day)'].max()) * 0.25 +
        (df['Data Usage (MB/day)'] / df['Data Usage (MB/day)'].max()) * 0.2
    )
    
    df['Battery_Efficiency'] = df['App Usage Time (min/day)'] / (df['Battery Drain (mAh/day)'] + 1)
    df['Data_Efficiency'] = df['App Usage Time (min/day)'] / (df['Data Usage (MB/day)'] + 1)
    
    # 3. Simulasi Rata-rata durasi tiap sesi (karena tidak ada data real session)
    df['Avg_Session_Duration'] = (df['Screen On Time (hours/day)'] * 60) / (df['Open Apps'] + 1)
    
    # 4. Simulasi Variansi harian (menggunakan statistical approximation)
    np.random.seed(42)
    df['Usage_Variance'] = (
        np.abs(df['App Usage Time (min/day)'] - df['App Usage Time (min/day)'].mean()) *
        (1 + np.random.normal(0, 0.1, len(df)))  # Add some realistic noise
    )
    
    df['Battery_Variance'] = (
        np.abs(df['Battery Drain (mAh/day)'] - df['Battery Drain (mAh/day)'].mean()) *
        (1 + np.random.normal(0, 0.1, len(df)))
    )
    
    # 5. Age-based features
    n_age_groups = 4
    age_labels = ['Children', 'Teenager', 'Adult', 'Elderly']
    age_values = df['Age'].values.reshape(-1, 1)
    kmeans_age = KMeans(n_clusters=n_age_groups, random_state=42, n_init=10)
    kmeans_age.fit(age_values)
    age_centers = np.sort(kmeans_age.cluster_centers_.flatten())
    # Buat batas antar cluster dari rata-rata centroid
    age_bins = [df['Age'].min()]
    age_bins += [np.mean([age_centers[i], age_centers[i+1]]) for i in range(len(age_centers)-1)]
    age_bins.append(df['Age'].max() + 1)
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
    
    # 6. Device and OS interaction features
    df['Device_OS_Combo'] = df['Device Model'].astype(str) + '_' + df['Operating System'].astype(str)
    
    # 7. Power User indicators
    df['Is_Power_User'] = (
        (df['App Usage Time (min/day)'] > df['App Usage Time (min/day)'].quantile(0.75)) &
        (df['Battery Drain (mAh/day)'] > df['Battery Drain (mAh/day)'].quantile(0.75))
    ).astype(int)
    
    # 8. Screen time categories
    n_screen_groups = 4
    screen_labels = ['Low', 'Medium', 'High', 'Extreme']
    screen_values = df['Screen On Time (hours/day)'].values.reshape(-1, 1)
    kmeans_screen = KMeans(n_clusters=n_screen_groups, random_state=42, n_init=10)
    kmeans_screen.fit(screen_values)
    screen_centers = np.sort(kmeans_screen.cluster_centers_.flatten())
    screen_bins = [df['Screen On Time (hours/day)'].min()]
    screen_bins += [np.mean([screen_centers[i], screen_centers[i+1]]) for i in range(len(screen_centers)-1)]
    screen_bins.append(df['Screen On Time (hours/day)'].max() + 1)
    df['Screen_Time_Category'] = pd.cut(df['Screen On Time (hours/day)'], bins=screen_bins, labels=screen_labels, include_lowest=True)
    
    # 9. Data usage categories
    n_data_groups = 4
    data_labels = ['Light', 'Moderate', 'Heavy', 'Extreme']
    data_values = df['Data Usage (MB/day)'].values.reshape(-1, 1)
    kmeans_data = KMeans(n_clusters=n_data_groups, random_state=42, n_init=10)
    kmeans_data.fit(data_values)
    data_centers = np.sort(kmeans_data.cluster_centers_.flatten())
    data_bins = [df['Data Usage (MB/day)'].min()]
    data_bins += [np.mean([data_centers[i], data_centers[i+1]]) for i in range(len(data_centers)-1)]
    data_bins.append(df['Data Usage (MB/day)'].max() + 1)
    df['Data_Usage_Category'] = pd.cut(df['Data Usage (MB/day)'], bins=data_bins, labels=data_labels, include_lowest=True)
    
    # 10. Polynomial features for key variables
    df['App_Usage_Squared'] = df['App Usage Time (min/day)'] ** 2
    df['Battery_Usage_Squared'] = df['Battery Drain (mAh/day)'] ** 2
    df['Screen_Time_Squared'] = df['Screen On Time (hours/day)'] ** 2
    
    # 11. Interaction features
    df['Age_Screen_Interaction'] = df['Age'] * df['Screen On Time (hours/day)']
    df['Battery_Data_Interaction'] = df['Battery Drain (mAh/day)'] * df['Data Usage (MB/day)']
    
    return df, age_bins, screen_bins, data_bins

def create_new_target_variable(df):
    print("Creating new target variable based on usage patterns...")
    
    # Method 1: Create target based on usage intensity clustering
    cluster_features = [
        'App Usage Time (min/day)', 'Battery Drain (mAh/day)', 'Screen On Time (hours/day)', 
        'Data Usage (MB/day)', 'Open Apps', 'Age'
    ]
    
    # Standardize features for clustering
    scaler_temp = StandardScaler()
    cluster_data = scaler_temp.fit_transform(df[cluster_features])
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Usage_Cluster'] = kmeans.fit_predict(cluster_data)
    
    # Method 2: Create target based on usage intensity quartiles
    df['Usage_Intensity_Target'] = pd.qcut(df['Usage_Intensity'], q=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
    
    # Method 3: Multi-dimensional target based on multiple factors
    screen_quartile = pd.qcut(df['Screen On Time (hours/day)'], q=4, labels=False)
    battery_quartile = pd.qcut(df['Battery Drain (mAh/day)'], q=4, labels=False)
    app_quartile = pd.qcut(df['App Usage Time (min/day)'], q=4, labels=False)
    open_apps_quartile = pd.qcut(df['Open Apps'], q=4, labels=False)
    age_quartile = pd.qcut(df['Age'], q=4, labels=False)
    
    # Create composite score
    df['Composite_Score'] = screen_quartile + battery_quartile + app_quartile + open_apps_quartile + age_quartile
    df['Multi_Dimensional_Target'] = pd.cut(df['Composite_Score'], bins=5, labels=['Low_User', 'Light_User', 'Moderate_User', 'Heavy_User', 'Power_User'])
    
    return df, kmeans

def perform_cluster_analysis(df):
    print("Performing cluster analysis...")
    
    # Features for clustering
    numeric_features = [
        'Age', 'App Usage Time (min/day)', 'Screen On Time (hours/day)',
        'Battery Drain (mAh/day)', 'Open Apps', 'Data Usage (MB/day)',
        'Usage_Intensity', 'Battery_Efficiency', 'Data_Efficiency',
        'Avg_Session_Duration', 'Usage_Variance'
    ]
    
    # Standardize data
    scaler = RobustScaler()  # More robust to outliers
    scaled_data = scaler.fit_transform(df[numeric_features])
    
    # Try different number of clusters
    inertias = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
    
    # Choose optimal number of clusters (elbow method - simplified) im using 5 clusters
    optimal_k = 5    

    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(scaled_data)
    df['Natural_Clusters'] = cluster_labels
    
    # Analyze clusters
    print(f"Cluster distribution: {np.bincount(cluster_labels)}")
    
    return df, final_kmeans, scaler

def preprocess_data_advanced():
    print("Loading dataset...")
    df = pd.read_csv('data/user_behavior_dataset.csv')
    
    print("Original dataset shape:", df.shape)
    print("Original columns:", df.columns.tolist())
    
    # Remove User ID but keep other potentially useful columns initially
    df.drop(['User ID'], axis=1, inplace=True)
    
    # Feature Engineering
    df, age_bins, screen_bins, data_bins = create_advanced_features(df)
    
    # Create new target variables
    df, usage_kmeans = create_new_target_variable(df)
    
    # Perform cluster analysis
    df, cluster_kmeans, cluster_scaler = perform_cluster_analysis(df)
    
    # Handle categorical variables with more sophisticated encoding
    print("\nProcessing categorical variables...")
    
    # One-hot encode categorical variables
    categorical_cols = [
        'Gender', 'Device Model', 'Operating System', 'Age_Group', 
        'Screen_Time_Category', 'Data_Usage_Category', 'Device_OS_Combo'
    ]
    
    # For high cardinality categorical variables, use frequency encoding
    high_cardinality_cols = ['Device Model', 'Operating System', 'Device_OS_Combo']
    
    for col in high_cardinality_cols:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[f'{col}_freq'] = df[col].map(freq_map)
    
    # One-hot encode remaining categorical variables
    remaining_categorical = [col for col in categorical_cols if col not in high_cardinality_cols]
    df_encoded = pd.get_dummies(df, columns=remaining_categorical, drop_first=True)
    
    # Drop high cardinality original columns
    df_encoded = df_encoded.drop(columns=high_cardinality_cols, errors='ignore')
    
    print(f"Features after encoding: {df_encoded.shape[1]} features")
    
    # Prepare different target options
    target_options = {
        'original': 'User Behavior Class',
        'usage_cluster': 'Usage_Cluster', 
        'intensity_target': 'Usage_Intensity_Target',
        'multi_dimensional': 'Multi_Dimensional_Target',
        'natural_clusters': 'Natural_Clusters'
    }
    
    # Process each target option
    processed_data = {}
    
    for target_name, target_col in target_options.items():
        print(f"\nProcessing target: {target_name}")
        
        if target_col not in df_encoded.columns:
            continue
            
        # Prepare features and target
        X = df_encoded.drop(columns=list(target_options.values()), errors='ignore')
        y = df_encoded[target_col]
        
        # Handle missing values in target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        # Encode target if categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = y.values
            label_encoder = None
        
        print(f"  Target classes: {np.unique(y_encoded)}")
        print(f"  Class distribution: {np.bincount(y_encoded)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        # Feature Selection
        print(f"  Applying feature selection...")
        
        # Use both statistical and mutual information based selection
        selector_f = SelectKBest(score_func=f_classif, k=30)
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=30)
        
        X_train_f = selector_f.fit_transform(X_train, y_train)
        X_train_mi = selector_mi.fit_transform(X_train, y_train)
        
        # Combine selected features
        selected_f = set(X.columns[selector_f.get_support()])
        selected_mi = set(X.columns[selector_mi.get_support()])
        selected_features = list(selected_f.union(selected_mi))
        
        # Ensure don't using too many features
        if len(selected_features) > 40:
            # Use intersection if union is too large
            selected_features = list(selected_f.intersection(selected_mi))
            if len(selected_features) < 20:
                selected_features = list(selected_f)[:25]
        
        age_related_features = [
            'Age', 'Age_Screen_Interaction', 'Age_Group_Adult',
            'Age_Group_Teenager', 'Age_Group_Children', 'Age_Group_Elderly'
        ]
        for f in age_related_features:
            if f in X.columns and f not in selected_features:
                selected_features.append(f)

        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        print(f"  Selected features ({len(selected_features)})")
        
        # Scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Store processed data
        processed_data[target_name] = {
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'X_train_unscaled': X_train_selected.values,
            'X_test_unscaled': X_test_selected.values,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'selected_features': selected_features,
            'feature_selector_f': selector_f,
            'feature_selector_mi': selector_mi
        }

        feature_stats = {
            'App Usage Time (min/day)_max': df['App Usage Time (min/day)'].max(),
            'App Usage Time (min/day)_mean': df['App Usage Time (min/day)'].mean(),
            'Battery Drain (mAh/day)_max': df['Battery Drain (mAh/day)'].max(),
            'Battery Drain (mAh/day)_mean': df['Battery Drain (mAh/day)'].mean(),
            'Screen On Time (hours/day)_max': df['Screen On Time (hours/day)'].max(),
            'Screen On Time (hours/day)_mean': df['Screen On Time (hours/day)'].mean(),
            'Data Usage (MB/day)_max': df['Data Usage (MB/day)'].max(),
            'Data Usage (MB/day)_mean': df['Data Usage (MB/day)'].mean(),
            'age_bins': age_bins,
            'screen_bins': screen_bins,
            'data_bins': data_bins
        }

    # Save all data
    print("\nSaving preprocessed data...")
    joblib.dump(processed_data, 'data/advanced_preprocessed_data.pkl')
    joblib.dump(feature_stats, 'models/feature_stats.pkl')
    
    print("Advanced preprocessing completed!")
    print(f"Available targets: {list(processed_data.keys())}")
    
    return processed_data

if __name__ == "__main__":
    processed_data = preprocess_data_advanced()