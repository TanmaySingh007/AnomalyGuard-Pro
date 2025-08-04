import pandas as pd
import numpy as np

def generate_synthetic_data(n_points=1000, anomaly_count=10):
    """
    Generate synthetic time-series data with controlled anomalies.
    
    Args:
        n_points (int): Number of data points to generate
        anomaly_count (int): Number of anomalies to introduce
    
    Returns:
        pd.DataFrame: DataFrame with timestamp and value columns
    """
    # Create a date range for the time-series
    start_date = pd.Timestamp('2024-01-01')
    end_date = start_date + pd.Timedelta(days=n_points-1)
    timestamps = pd.date_range(start=start_date, end=end_date, periods=n_points)
    
    # Generate base time-series with a sine wave pattern
    # This creates a regular oscillating pattern that's common in many real-world scenarios
    time_values = np.linspace(0, 4*np.pi, n_points)
    base_values = np.sin(time_values) * 10  # Amplitude of 10
    
    # Add random noise to make it more realistic
    # This simulates measurement errors and natural variations
    noise = np.random.normal(0, 0.5, n_points)
    values = base_values + noise
    
    # Introduce anomalies by selecting random indices
    # We'll add both positive spikes and negative dips to simulate different types of anomalies
    anomaly_indices = np.random.choice(n_points, size=anomaly_count, replace=False)
    
    for idx in anomaly_indices:
        # Randomly decide if this should be a spike (positive) or dip (negative)
        if np.random.random() > 0.5:
            # Add a positive spike (anomaly above normal range)
            values[idx] += np.random.uniform(8, 15)
        else:
            # Add a negative dip (anomaly below normal range)
            values[idx] -= np.random.uniform(8, 15)
    
    # Create DataFrame with timestamp and value columns
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    return df

def main():
    """
    Main function to generate and save synthetic data.
    """
    print("Generating synthetic time-series data...")
    
    # Generate the synthetic dataset
    df = generate_synthetic_data(n_points=1000, anomaly_count=10)
    
    # Save to CSV file
    output_file = 'simulated_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Data generated successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Data saved to: {output_file}")
    print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

if __name__ == "__main__":
    main() 