# ./med_risk_pred/data/bed_occupancy_processor.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

class BedDataProcessor:
    def __init__(self, config_path='configs/base.yaml'):
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        
    def _load_config(self, path):
        # Implementation for loading YAML config
        return {
            'window_size': 24,
            'test_size': 0.2,
            'features': ['risk_score', 'current_occupancy', 'admissions', 'discharges', 'hour', 'day_of_week']
        }

    def process_raw_data(self, df):
        """Process raw bed occupancy data"""
        # Time-based features
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Create rolling features
        df['rolling_occupancy_6h'] = df['current_occupancy'].rolling(window=6).mean()
        df['rolling_admissions_12h'] = df['admissions'].rolling(window=12).sum()
        
        # Handle missing values
        df = df.fillna(method='ffill').dropna()
        
        # Select features from config
        features = self.config['features'] + ['rolling_occupancy_6h', 'rolling_admissions_12h']
        target = 'future_occupancy'
        
        return df[features], df[target]

    def temporal_split(self, X, y):
        """Time-based train-test split"""
        split_index = int(len(X) * (1 - self.config['test_size']))
        return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    def scale_features(self, X_train, X_test):
        """Scale features using stored scaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def generate_sequences(self, X, y, window_size=24):
        """Create time series sequences"""
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size):
            X_seq.append(X[i:i+window_size])
            y_seq.append(y[i+window_size])
        return np.array(X_seq), np.array(y_seq)
