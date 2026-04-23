from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import xgboost as xgb
import joblib
import os

class F1AdaptivePredictor:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.base_rf = None
        self.adaptive_xgb = None
        self.recent_buffer = []
        self.update_counter = 0

    def train_initial(self, X, y):
        """После полного сезона заново обучаем базовую модель RandomForest"""
        self.base_rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.base_rf.fit(X, y)

        self.adaptive_xgb = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
        self.adaptive_xgb.fit(X, y)

    def add_race_and_update(self, X_race, y_race):
        """Дообучение после каждой гонки"""
        self.update_counter += 1
        self.recent_buffer.append((X_race, y_race))
        if len(self.recent_buffer) > 10:
            self.recent_buffer.pop(0)

        X_buffer = pd.concat([x for x, _ in self.recent_buffer], ignore_index=True)
        y_buffer = pd.concat([y for _, y in self.recent_buffer], ignore_index=True)

        self.adaptive_xgb = xgb.XGBRegressor(
            n_estimators=20,
            max_depth=4,
            learning_rate=0.01,
            random_state=42
        )
        self.adaptive_xgb.fit(
            X_buffer, y_buffer,
            xgb_model=self.adaptive_xgb
        )

    def predict(self, X):
        """Предсказание: 60% RF + 40% XGBoost"""
        pred_rf = self.base_rf.predict(X) * 0.6
        pred_xgb = self.adaptive_xgb.predict(X) * 0.4

        return pred_rf + pred_xgb