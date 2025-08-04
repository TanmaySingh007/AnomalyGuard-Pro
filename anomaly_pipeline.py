import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path='simulated_data.csv'):
    """
    Load the simulated time-series data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing the data
    
    Returns:
        pd.DataFrame: Loaded DataFrame with timestamp and value columns
    """
    try:
        df = pd.read_csv(file_path)
        # Convert timestamp column to datetime if it's not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Data loaded successfully from {file_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Please run data_simulator.py first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_features(df):
    """
    Prepare features for anomaly detection with advanced feature engineering.
    
    Args:
        df (pd.DataFrame): Input DataFrame with timestamp and value columns
    
    Returns:
        tuple: (features, scaler) - Prepared features and fitted scaler
    """
    # Extract the value column as our main feature
    values = df['value'].values.reshape(-1, 1)
    
    # Create additional time-based features for more robust anomaly detection
    # These features can help the model understand temporal patterns
    df_features = df.copy()
    
    # Add comprehensive time-based features
    df_features['hour'] = df_features['timestamp'].dt.hour
    df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
    df_features['day_of_month'] = df_features['timestamp'].dt.day
    df_features['month'] = df_features['timestamp'].dt.month
    df_features['quarter'] = df_features['timestamp'].dt.quarter
    df_features['is_weekend'] = df_features['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    df_features['is_business_hour'] = ((df_features['hour'] >= 9) & (df_features['hour'] <= 17)).astype(int)
    
    # Add cyclical time features (sine and cosine transformations)
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Add multiple rolling statistics with different window sizes
    window_sizes = [5, 10, 20, 50]
    for window in window_sizes:
        df_features[f'rolling_mean_{window}'] = df_features['value'].rolling(window=window, center=True).mean()
        df_features[f'rolling_std_{window}'] = df_features['value'].rolling(window=window, center=True).std()
        df_features[f'rolling_min_{window}'] = df_features['value'].rolling(window=window, center=True).min()
        df_features[f'rolling_max_{window}'] = df_features['value'].rolling(window=window, center=True).max()
        df_features[f'rolling_median_{window}'] = df_features['value'].rolling(window=window, center=True).median()
    
    # Add lag features (previous values)
    for lag in [1, 2, 3, 5, 10]:
        df_features[f'lag_{lag}'] = df_features['value'].shift(lag)
        df_features[f'diff_{lag}'] = df_features['value'] - df_features[f'lag_{lag}']
        df_features[f'pct_change_{lag}'] = df_features['value'].pct_change(lag)
    
    # Add statistical features
    df_features['z_score'] = (df_features['value'] - df_features['value'].rolling(window=20, center=True).mean()) / \
                             df_features['value'].rolling(window=20, center=True).std()
    
    # Add volatility features
    df_features['volatility'] = df_features['value'].rolling(window=20, center=True).std() / \
                               df_features['value'].rolling(window=20, center=True).mean()
    
    # Add trend features
    df_features['trend'] = df_features['value'].rolling(window=20, center=True).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    
    # Add seasonality features (if data has seasonal patterns)
    df_features['seasonal_factor'] = df_features['value'].rolling(window=24, center=True).mean() - \
                                    df_features['value'].rolling(window=168, center=True).mean()  # 24h vs 7 days
    
    # Add outlier indicators
    q1 = df_features['value'].rolling(window=50, center=True).quantile(0.25)
    q3 = df_features['value'].rolling(window=50, center=True).quantile(0.75)
    iqr = q3 - q1
    df_features['is_outlier_iqr'] = ((df_features['value'] < (q1 - 1.5 * iqr)) | 
                                     (df_features['value'] > (q3 + 1.5 * iqr))).astype(int)
    
    # Add rate of change features
    df_features['rate_of_change'] = df_features['value'].diff() / df_features['value'].shift(1)
    df_features['acceleration'] = df_features['rate_of_change'].diff()
    
    # Fill NaN values from rolling calculations
    df_features = df_features.fillna(method='bfill').fillna(method='ffill')
    
    # Select comprehensive feature set for the model
    feature_columns = [
        'value', 'hour', 'day_of_week', 'month', 'quarter', 'is_weekend', 'is_business_hour',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'rolling_mean_5', 'rolling_std_5', 'rolling_min_5', 'rolling_max_5', 'rolling_median_5',
        'rolling_mean_10', 'rolling_std_10', 'rolling_min_10', 'rolling_max_10', 'rolling_median_10',
        'rolling_mean_20', 'rolling_std_20', 'rolling_min_20', 'rolling_max_20', 'rolling_median_20',
        'lag_1', 'lag_2', 'lag_3', 'diff_1', 'diff_2', 'diff_3', 'pct_change_1', 'pct_change_2',
        'z_score', 'volatility', 'trend', 'seasonal_factor', 'is_outlier_iqr',
        'rate_of_change', 'acceleration'
    ]
    
    # Remove any remaining NaN values
    df_features = df_features.dropna()
    
    features = df_features[feature_columns].values
    
    # Standardize the features (important for Isolation Forest)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    print(f"Advanced features prepared: {features.shape}")
    print(f"Number of feature columns: {len(feature_columns)}")
    print(f"Feature categories:")
    print(f"  - Time-based: {sum(1 for f in feature_columns if any(x in f for x in ['hour', 'day', 'month', 'weekend', 'business']))}")
    print(f"  - Cyclical: {sum(1 for f in feature_columns if 'sin' in f or 'cos' in f)}")
    print(f"  - Rolling statistics: {sum(1 for f in feature_columns if 'rolling' in f)}")
    print(f"  - Lag features: {sum(1 for f in feature_columns if 'lag' in f or 'diff' in f or 'pct_change' in f)}")
    print(f"  - Statistical: {sum(1 for f in feature_columns if f in ['z_score', 'volatility', 'trend', 'seasonal_factor', 'is_outlier_iqr', 'rate_of_change', 'acceleration'])}")
    print(f"Feature statistics - Mean: {features.mean():.3f}, Std: {features.std():.3f}")
    
    return features, scaler

def prepare_simple_features(df):
    """
    Prepare simple features using only the value column (for comparison).
    
    Args:
        df (pd.DataFrame): Input DataFrame with timestamp and value columns
    
    Returns:
        tuple: (features, scaler) - Prepared features and fitted scaler
    """
    # Extract only the value column as feature (2D array as required by Isolation Forest)
    values = df['value'].values.reshape(-1, 1)
    
    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(values)
    
    print(f"Simple features prepared: {features.shape}")
    print(f"Feature statistics - Mean: {features.mean():.3f}, Std: {features.std():.3f}")
    
    return features, scaler

def train_isolation_forest(features, contamination='auto', random_state=42):
    """
    Train an Isolation Forest model for anomaly detection.
    
    Args:
        features (np.array): Prepared feature array
        contamination (str or float): Expected proportion of anomalies ('auto' or float)
        random_state (int): Random seed for reproducibility
    
    Returns:
        IsolationForest: Trained Isolation Forest model
    """
    # Initialize Isolation Forest
    # Isolation Forest works by isolating observations by randomly selecting a feature
    # and then randomly selecting a split value between the max and min values
    iso_forest = IsolationForest(
        contamination=contamination,  # 'auto' for automatic detection or float for manual setting
        random_state=random_state,
        n_estimators=100,  # Number of trees in the forest
        max_samples='auto',  # Number of samples to draw for each tree
        bootstrap=True,  # Use bootstrap sampling for better robustness
        max_features=1.0  # Use all features for each split
    )
    
    # Train the model
    print("Training Isolation Forest model...")
    print(f"Training data shape: {features.shape}")
    print(f"Using contamination: {contamination}")
    print(f"Number of estimators: 100")
    
    iso_forest.fit(features)
    
    # Print model information
    print(f"Model training completed successfully!")
    print(f"Model parameters:")
    print(f"  - Contamination: {iso_forest.contamination}")
    print(f"  - Number of estimators: {iso_forest.n_estimators}")
    print(f"  - Max samples: {iso_forest.max_samples}")
    
    return iso_forest

def train_multiple_models(features, contamination_values=['auto', 0.05, 0.1, 0.15]):
    """
    Train multiple Isolation Forest models with different contamination values for comparison.
    
    Args:
        features (np.array): Prepared feature array
        contamination_values (list): List of contamination values to try
    
    Returns:
        dict: Dictionary of trained models with contamination values as keys
    """
    models = {}
    
    print("Training multiple models for comparison...")
    print("=" * 50)
    
    for contamination in contamination_values:
        print(f"\nTraining model with contamination={contamination}")
        model = train_isolation_forest(features, contamination=contamination)
        models[contamination] = model
        
        # Quick evaluation
        predictions = model.predict(features)
        n_anomalies = (predictions == -1).sum()
        anomaly_percentage = (n_anomalies / len(features)) * 100
        print(f"  Detected anomalies: {n_anomalies} ({anomaly_percentage:.2f}%)")
    
    print("\n" + "=" * 50)
    print("Model training comparison completed!")
    
    return models

def analyze_feature_importance(model, feature_names, X):
    """
    Analyze feature importance for the Isolation Forest model.
    
    Args:
        model (IsolationForest): Trained Isolation Forest model
        feature_names (list): List of feature names
        X (np.array): Feature matrix
    
    Returns:
        dict: Dictionary with feature importance scores
    """
    # Calculate feature importance based on anomaly score changes
    base_scores = model.decision_function(X)
    feature_importance = {}
    
    for i, feature_name in enumerate(feature_names):
        # Create perturbed data by adding noise to one feature
        X_perturbed = X.copy()
        X_perturbed[:, i] += np.random.normal(0, 0.1, X_perturbed.shape[0])
        
        # Calculate new scores
        perturbed_scores = model.decision_function(X_perturbed)
        
        # Feature importance is the average change in anomaly scores
        importance = np.mean(np.abs(perturbed_scores - base_scores))
        feature_importance[feature_name] = importance
    
    # Sort by importance
    sorted_importance = dict(sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True))
    
    return sorted_importance

def evaluate_model_performance(df_with_predictions):
    """
    Evaluate the performance of the anomaly detection model.
    
    Args:
        df_with_predictions (pd.DataFrame): DataFrame with predictions
    
    Returns:
        dict: Performance metrics
    """
    # Calculate basic statistics
    total_points = len(df_with_predictions)
    n_anomalies = df_with_predictions['is_anomaly'].sum()
    n_normal = total_points - n_anomalies
    
    # Calculate anomaly score statistics
    anomaly_scores = df_with_predictions['anomaly_score']
    normal_scores = df_with_predictions[~df_with_predictions['is_anomaly']]['anomaly_score']
    anomaly_only_scores = df_with_predictions[df_with_predictions['is_anomaly']]['anomaly_score']
    
    # Performance metrics
    metrics = {
        'total_points': total_points,
        'anomalies_detected': n_anomalies,
        'normal_points': n_normal,
        'anomaly_percentage': (n_anomalies / total_points) * 100,
        'mean_anomaly_score': anomaly_scores.mean(),
        'std_anomaly_score': anomaly_scores.std(),
        'min_anomaly_score': anomaly_scores.min(),
        'max_anomaly_score': anomaly_scores.max(),
        'normal_score_mean': normal_scores.mean() if len(normal_scores) > 0 else 0,
        'anomaly_score_mean': anomaly_only_scores.mean() if len(anomaly_only_scores) > 0 else 0,
        'score_separation': abs(normal_scores.mean() - anomaly_only_scores.mean()) if len(normal_scores) > 0 and len(anomaly_only_scores) > 0 else 0
    }
    
    # Print performance summary
    print(f"\n{'='*60}")
    print(f"MODEL PERFORMANCE EVALUATION")
    print(f"{'='*60}")
    print(f"Total data points: {metrics['total_points']}")
    print(f"Anomalies detected: {metrics['anomalies_detected']} ({metrics['anomaly_percentage']:.2f}%)")
    print(f"Normal points: {metrics['normal_points']}")
    print(f"\nAnomaly Score Statistics:")
    print(f"  Mean: {metrics['mean_anomaly_score']:.4f}")
    print(f"  Std: {metrics['std_anomaly_score']:.4f}")
    print(f"  Range: {metrics['min_anomaly_score']:.4f} to {metrics['max_anomaly_score']:.4f}")
    print(f"\nScore Separation:")
    print(f"  Normal points mean: {metrics['normal_score_mean']:.4f}")
    print(f"  Anomaly points mean: {metrics['anomaly_score_mean']:.4f}")
    print(f"  Separation: {metrics['score_separation']:.4f}")
    
    return metrics

def detect_anomalies(model, features, df):
    """
    Detect anomalies using the trained Isolation Forest model.
    
    Args:
        model (IsolationForest): Trained Isolation Forest model
        features (np.array): Prepared feature array
        df (pd.DataFrame): Original DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with anomaly predictions added
    """
    # Predict anomalies (-1 for anomalies, 1 for normal points)
    predictions = model.predict(features)
    
    # Add predictions to the DataFrame
    df_with_predictions = df.copy()
    df_with_predictions['anomaly_prediction'] = predictions  # Renamed for clarity
    df_with_predictions['is_anomaly'] = (predictions == -1)
    
    # Calculate anomaly scores (lower scores indicate more anomalous)
    anomaly_scores = model.decision_function(features)
    df_with_predictions['anomaly_score'] = anomaly_scores
    
    # Print summary statistics
    n_anomalies = df_with_predictions['is_anomaly'].sum()
    total_points = len(df_with_predictions)
    anomaly_percentage = (n_anomalies / total_points) * 100
    
    print(f"\nAnomaly Detection Results:")
    print(f"Total data points: {total_points}")
    print(f"Detected anomalies: {n_anomalies}")
    print(f"Anomaly percentage: {anomaly_percentage:.2f}%")
    print(f"Anomaly score range: {anomaly_scores.min():.3f} to {anomaly_scores.max():.3f}")
    
    return df_with_predictions

def identify_anomalies(df_with_predictions):
    """
    Identify and display detailed information about detected anomalies.
    
    Args:
        df_with_predictions (pd.DataFrame): DataFrame with anomaly predictions
    
    Returns:
        pd.DataFrame: DataFrame containing only the detected anomalies
    """
    # Filter DataFrame to get only anomalies (where anomaly_prediction is -1)
    anomalies_df = df_with_predictions[df_with_predictions['anomaly_prediction'] == -1].copy()
    
    if anomalies_df.empty:
        print("\nNo anomalies detected in the dataset.")
        return anomalies_df
    
    # Sort by anomaly score (most anomalous first)
    anomalies_df = anomalies_df.sort_values('anomaly_score', ascending=True)
    
    print(f"\n{'='*60}")
    print(f"DETECTED ANOMALIES DETAILS")
    print(f"{'='*60}")
    print(f"Total anomalies found: {len(anomalies_df)}")
    print(f"{'='*60}")
    
    # Display detailed information for each anomaly
    for idx, (_, anomaly) in enumerate(anomalies_df.iterrows(), 1):
        print(f"\nAnomaly #{idx}:")
        print(f"  Timestamp: {anomaly['timestamp']}")
        print(f"  Value: {anomaly['value']:.4f}")
        print(f"  Anomaly Score: {anomaly['anomaly_score']:.4f}")
        print(f"  Prediction: {anomaly['anomaly_prediction']}")
        
        # Add context about the anomaly
        if anomaly['anomaly_score'] < -0.5:
            severity = "HIGH"
        elif anomaly['anomaly_score'] < -0.2:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        print(f"  Severity: {severity}")
        print(f"  {'-'*40}")
    
    # Print summary statistics
    print(f"\nAnomaly Summary Statistics:")
    print(f"  Average anomaly score: {anomalies_df['anomaly_score'].mean():.4f}")
    print(f"  Min anomaly score: {anomalies_df['anomaly_score'].min():.4f}")
    print(f"  Max anomaly score: {anomalies_df['anomaly_score'].max():.4f}")
    print(f"  Value range of anomalies: {anomalies_df['value'].min():.4f} to {anomalies_df['value'].max():.4f}")
    
    return anomalies_df

def visualize_results(df_with_predictions, save_plot=True):
    """
    Visualize the time-series data with detected anomalies highlighted.
    
    Args:
        df_with_predictions (pd.DataFrame): DataFrame with anomaly predictions
        save_plot (bool): Whether to save the plot to file
    """
    # Create a figure with subplots for comprehensive visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Time-series with anomalies highlighted
    # Plot normal data points
    normal_data = df_with_predictions[~df_with_predictions['is_anomaly']]
    ax1.plot(normal_data['timestamp'], normal_data['value'], 
             color='blue', alpha=0.7, linewidth=1, label='Normal Data Points')
    
    # Highlight anomalies with distinct markers
    anomalies = df_with_predictions[df_with_predictions['is_anomaly']]
    if not anomalies.empty:
        # Use different colors based on severity
        high_severity = anomalies[anomalies['anomaly_score'] < -0.5]
        medium_severity = anomalies[(anomalies['anomaly_score'] >= -0.5) & (anomalies['anomaly_score'] < -0.2)]
        low_severity = anomalies[anomalies['anomaly_score'] >= -0.2]
        
        if not high_severity.empty:
            ax1.scatter(high_severity['timestamp'], high_severity['value'], 
                       color='red', s=100, marker='o', alpha=0.9, 
                       label='High Severity Anomalies', edgecolors='black', linewidth=1)
        
        if not medium_severity.empty:
            ax1.scatter(medium_severity['timestamp'], medium_severity['value'], 
                       color='orange', s=80, marker='s', alpha=0.8, 
                       label='Medium Severity Anomalies', edgecolors='black', linewidth=1)
        
        if not low_severity.empty:
            ax1.scatter(low_severity['timestamp'], low_severity['value'], 
                       color='yellow', s=60, marker='^', alpha=0.7, 
                       label='Low Severity Anomalies', edgecolors='black', linewidth=1)
    
    ax1.set_title('Time-Series Anomaly Detection Results', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Timestamp', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Anomaly scores over time
    ax2.plot(df_with_predictions['timestamp'], df_with_predictions['anomaly_score'], 
             color='green', alpha=0.7, linewidth=1, label='Anomaly Score')
    
    # Add threshold lines
    ax2.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7, label='High Severity Threshold')
    ax2.axhline(y=-0.2, color='orange', linestyle='--', alpha=0.7, label='Medium Severity Threshold')
    ax2.axhline(y=0, color='blue', linestyle='--', alpha=0.7, label='Normal Threshold')
    
    # Highlight anomaly regions
    anomaly_regions = df_with_predictions[df_with_predictions['is_anomaly']]
    if not anomaly_regions.empty:
        ax2.scatter(anomaly_regions['timestamp'], anomaly_regions['anomaly_score'], 
                   color='red', s=30, alpha=0.8, label='Anomaly Points')
    
    ax2.set_title('Anomaly Scores Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Timestamp', fontsize=12)
    ax2.set_ylabel('Anomaly Score', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'anomaly_detection_results.png'")
    
    plt.show()

def create_additional_visualizations(df_with_predictions, save_plots=True):
    """
    Create additional visualizations for deeper analysis.
    
    Args:
        df_with_predictions (pd.DataFrame): DataFrame with anomaly predictions
        save_plots (bool): Whether to save the plots to files
    """
    # Create a comprehensive analysis figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Value distribution with anomalies highlighted
    normal_values = df_with_predictions[~df_with_predictions['is_anomaly']]['value']
    anomaly_values = df_with_predictions[df_with_predictions['is_anomaly']]['value']
    
    ax1.hist(normal_values, bins=30, alpha=0.7, color='blue', label='Normal Data')
    if not anomaly_values.empty:
        ax1.hist(anomaly_values, bins=15, alpha=0.8, color='red', label='Anomalies')
    
    ax1.set_title('Value Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Anomaly score distribution
    ax2.hist(df_with_predictions['anomaly_score'], bins=30, alpha=0.7, color='green')
    ax2.axvline(x=-0.5, color='red', linestyle='--', alpha=0.7, label='High Severity')
    ax2.axvline(x=-0.2, color='orange', linestyle='--', alpha=0.7, label='Medium Severity')
    ax2.set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Anomaly Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time of day analysis
    df_with_predictions['hour'] = df_with_predictions['timestamp'].dt.hour
    hourly_anomalies = df_with_predictions[df_with_predictions['is_anomaly']].groupby('hour').size()
    hourly_total = df_with_predictions.groupby('hour').size()
    anomaly_rate = (hourly_anomalies / hourly_total * 100).fillna(0)
    
    ax3.bar(anomaly_rate.index, anomaly_rate.values, alpha=0.7, color='orange')
    ax3.set_title('Anomaly Rate by Hour of Day', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Hour of Day', fontsize=12)
    ax3.set_ylabel('Anomaly Rate (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rolling statistics
    window_size = 20
    rolling_mean = df_with_predictions['value'].rolling(window=window_size, center=True).mean()
    rolling_std = df_with_predictions['value'].rolling(window=window_size, center=True).std()
    
    ax4.plot(df_with_predictions['timestamp'], rolling_mean, color='blue', alpha=0.7, label='Rolling Mean')
    ax4.fill_between(df_with_predictions['timestamp'], 
                     rolling_mean - 2*rolling_std, 
                     rolling_mean + 2*rolling_std, 
                     alpha=0.3, color='blue', label='±2σ Range')
    
    # Highlight anomalies on rolling plot
    anomalies = df_with_predictions[df_with_predictions['is_anomaly']]
    if not anomalies.empty:
        ax4.scatter(anomalies['timestamp'], anomalies['value'], 
                   color='red', s=50, alpha=0.8, label='Anomalies')
    
    ax4.set_title(f'Rolling Statistics (Window={window_size})', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Timestamp', fontsize=12)
    ax4.set_ylabel('Value', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('anomaly_analysis_detailed.png', dpi=300, bbox_inches='tight')
        print("Detailed analysis plot saved as 'anomaly_analysis_detailed.png'")
    
    plt.show()

def create_feature_analysis_plots(df_with_predictions, feature_importance=None):
    """
    Create advanced feature analysis plots.
    
    Args:
        df_with_predictions (pd.DataFrame): DataFrame with predictions
        feature_importance (dict): Feature importance dictionary
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Feature importance (if available)
    if feature_importance:
        top_features = list(feature_importance.keys())[:10]
        top_importance = list(feature_importance.values())[:10]
        
        ax1.barh(range(len(top_features)), top_importance, color='steelblue')
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features)
        ax1.set_xlabel('Importance Score')
        ax1.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Anomaly score distribution by time of day
    df_with_predictions['hour'] = df_with_predictions['timestamp'].dt.hour
    hourly_scores = df_with_predictions.groupby('hour')['anomaly_score'].mean()
    
    ax2.plot(hourly_scores.index, hourly_scores.values, marker='o', linewidth=2, markersize=6)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average Anomaly Score')
    ax2.set_title('Anomaly Scores by Hour of Day', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))
    
    # Plot 3: Anomaly score vs value scatter plot
    normal_points = df_with_predictions[~df_with_predictions['is_anomaly']]
    anomaly_points = df_with_predictions[df_with_predictions['is_anomaly']]
    
    ax3.scatter(normal_points['value'], normal_points['anomaly_score'], 
                alpha=0.6, color='blue', s=20, label='Normal')
    if not anomaly_points.empty:
        ax3.scatter(anomaly_points['value'], anomaly_points['anomaly_score'], 
                   alpha=0.8, color='red', s=40, label='Anomalies')
    
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Anomaly Score')
    ax3.set_title('Anomaly Score vs Value', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rolling statistics with anomalies
    window_size = 20
    rolling_mean = df_with_predictions['value'].rolling(window=window_size, center=True).mean()
    rolling_std = df_with_predictions['value'].rolling(window=window_size, center=True).std()
    
    ax4.plot(df_with_predictions['timestamp'], rolling_mean, color='blue', alpha=0.7, label='Rolling Mean')
    ax4.fill_between(df_with_predictions['timestamp'], 
                     rolling_mean - 2*rolling_std, 
                     rolling_mean + 2*rolling_std, 
                     alpha=0.3, color='blue', label='±2σ Range')
    
    # Highlight anomalies
    anomalies = df_with_predictions[df_with_predictions['is_anomaly']]
    if not anomalies.empty:
        ax4.scatter(anomalies['timestamp'], anomalies['value'], 
                   color='red', s=50, alpha=0.8, label='Anomalies')
    
    ax4.set_title(f'Rolling Statistics with Anomalies (Window={window_size})', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Timestamp')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('feature_analysis_plots.png', dpi=300, bbox_inches='tight')
    print("Feature analysis plots saved as 'feature_analysis_plots.png'")
    plt.show()

def generate_anomaly_report(df_with_predictions, metrics, feature_importance=None):
    """
    Generate a comprehensive anomaly detection report.
    
    Args:
        df_with_predictions (pd.DataFrame): DataFrame with predictions
        metrics (dict): Performance metrics
        feature_importance (dict): Feature importance dictionary
    """
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE ANOMALY DETECTION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 40)
    report.append(f"Total data points analyzed: {metrics['total_points']}")
    report.append(f"Anomalies detected: {metrics['anomalies_detected']} ({metrics['anomaly_percentage']:.2f}%)")
    report.append(f"Model separation score: {metrics['score_separation']:.4f}")
    report.append("")
    
    # Anomaly Details
    anomalies = df_with_predictions[df_with_predictions['is_anomaly']]
    if not anomalies.empty:
        report.append("ANOMALY DETAILS")
        report.append("-" * 40)
        report.append(f"Most severe anomaly score: {anomalies['anomaly_score'].min():.4f}")
        report.append(f"Average anomaly score: {anomalies['anomaly_score'].mean():.4f}")
        report.append(f"Anomaly value range: {anomalies['value'].min():.4f} to {anomalies['value'].max():.4f}")
        report.append("")
        
        # Temporal analysis
        report.append("TEMPORAL ANALYSIS")
        report.append("-" * 40)
        hourly_anomalies = anomalies.groupby(anomalies['timestamp'].dt.hour).size()
        report.append("Anomalies by hour of day:")
        for hour, count in hourly_anomalies.items():
            report.append(f"  Hour {hour:02d}: {count} anomalies")
        report.append("")
    
    # Feature Importance (if available)
    if feature_importance:
        report.append("TOP FEATURE IMPORTANCE")
        report.append("-" * 40)
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
            report.append(f"{i:2d}. {feature}: {importance:.4f}")
        report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 40)
    if metrics['anomaly_percentage'] > 10:
        report.append("⚠️  High anomaly rate detected - consider investigating data quality")
    elif metrics['anomaly_percentage'] > 5:
        report.append("⚠️  Moderate anomaly rate - monitor closely")
    else:
        report.append("✅ Normal anomaly rate detected")
    
    if metrics['score_separation'] > 0.5:
        report.append("✅ Good separation between normal and anomalous points")
    else:
        report.append("⚠️  Low separation - consider feature engineering improvements")
    
    report.append("")
    report.append("=" * 80)
    
    # Save report to file
    with open('anomaly_detection_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # Print report
    print('\n'.join(report))
    print(f"\nReport saved to 'anomaly_detection_report.txt'")

def main():
    """
    Main function to run the complete anomaly detection pipeline.
    """
    print("=== Automated Data Anomaly Detection Pipeline ===\n")
    
    # Step 1: Load the data
    print("Step 1: Loading data...")
    df = load_data()
    if df is None:
        return
    
    # Step 2: Prepare features
    print("\nStep 2: Preparing features...")
    # Use enhanced features by default (can be changed to prepare_simple_features for basic approach)
    features, scaler = prepare_features(df)
    
    # Get feature names for analysis
    feature_names = [
        'value', 'hour', 'day_of_week', 'month', 'quarter', 'is_weekend', 'is_business_hour',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'rolling_mean_5', 'rolling_std_5', 'rolling_min_5', 'rolling_max_5', 'rolling_median_5',
        'rolling_mean_10', 'rolling_std_10', 'rolling_min_10', 'rolling_max_10', 'rolling_median_10',
        'rolling_mean_20', 'rolling_std_20', 'rolling_min_20', 'rolling_max_20', 'rolling_median_20',
        'lag_1', 'lag_2', 'lag_3', 'diff_1', 'diff_2', 'diff_3', 'pct_change_1', 'pct_change_2',
        'z_score', 'volatility', 'trend', 'seasonal_factor', 'is_outlier_iqr',
        'rate_of_change', 'acceleration'
    ]
    
    # Step 3: Train Isolation Forest model
    print("\nStep 3: Training anomaly detection model...")
    # Use automatic contamination detection for better results
    model = train_isolation_forest(features, contamination='auto')
    
    # Step 3.5: Analyze feature importance
    print("\nStep 3.5: Analyzing feature importance...")
    feature_importance = analyze_feature_importance(model, feature_names, features)
    print("Top 5 most important features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:5], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Optional: Uncomment the line below to train multiple models for comparison
    # models = train_multiple_models(features)
    # model = models['auto']  # Use the auto model for further processing
    
    # Step 4: Detect anomalies
    print("\nStep 4: Detecting anomalies...")
    df_with_predictions = detect_anomalies(model, features, df)
    
    # Step 4.5: Identify and display detailed anomaly information
    print("\nStep 4.5: Identifying anomalies...")
    anomalies_df = identify_anomalies(df_with_predictions)
    
    # Step 4.6: Evaluate model performance
    print("\nStep 4.6: Evaluating model performance...")
    metrics = evaluate_model_performance(df_with_predictions)
    
    # Step 5: Visualize results
    print("\nStep 5: Visualizing results...")
    visualize_results(df_with_predictions)
    
    # Step 5.5: Create additional detailed visualizations
    print("\nStep 5.5: Creating additional analysis plots...")
    create_additional_visualizations(df_with_predictions)
    
    # Step 5.6: Create feature analysis plots
    print("\nStep 5.6: Creating feature analysis plots...")
    create_feature_analysis_plots(df_with_predictions, feature_importance)
    
    # Step 5.7: Generate comprehensive report
    print("\nStep 5.7: Generating comprehensive report...")
    generate_anomaly_report(df_with_predictions, metrics, feature_importance)
    
    # Save results to CSV
    output_file = 'anomaly_detection_results.csv'
    df_with_predictions.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Display final summary
    if not anomalies_df.empty:
        print(f"\nFinal Summary:")
        print(f"  - Total anomalies detected: {len(anomalies_df)}")
        print(f"  - Most severe anomaly score: {anomalies_df['anomaly_score'].min():.4f}")
        print(f"  - Anomalies saved to CSV with full details")
    else:
        print(f"\nNo anomalies were detected in the dataset.")
    
    print("\n=== Pipeline completed successfully! ===")

if __name__ == "__main__":
    main() 