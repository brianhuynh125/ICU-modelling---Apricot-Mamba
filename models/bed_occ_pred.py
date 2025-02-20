# ./med_risk_pred/models/bed_occupancy_predictor.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class BedOccupancyPredictor:
    def __init__(self, model_type='lstm'):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=200, max_depth=10),
            'xgboost': XGBRegressor(n_estimators=150, learning_rate=0.05),
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5),
            'lstm': self._build_lstm_model()
        }
        self.selected_model = self.models[model_type]
    def skip():
        return 0   
    def _build_lstm_model(self):
        model = Sequential([
            LSTM(64, input_shape=(None, 8), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, y_train, X_test=None, y_test=None):
        if isinstance(self.selected_model, tf.keras.Model):
            self.selected_model.fit(X_train, y_train, 
                                   epochs=100, batch_size=32,
                                   validation_data=(X_test, y_test),
                                   verbose=0)
        else:
            self.selected_model.fit(X_train, y_train)

    def predict(self, X):
        if isinstance(self.selected_model, tf.keras.Model):
            return self.selected_model.predict(X).flatten()
        return self.selected_model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return {
            'mae': mean_absolute_error(y, preds),
            'mse': mean_squared_error(y, preds),
            'r2': r2_score(y, preds)
        }

    @staticmethod
    def compare_models(X_train, y_train, X_test, y_test):
        results = {}
        for model_name in ['random_forest', 'xgboost', 'svr', 'gradient_boosting', 'lstm']:
            predictor = BedOccupancyPredictor(model_name)
            predictor.train(X_train, y_train, X_test, y_test)
            results[model_name] = predictor.evaluate(X_test, y_test)
        return results
