from pyexpat import features
from sklearn.ensemble import RandomForestRegressor
import os
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_or_create_xgboost(model_path, feature_columns):
    """Загружает XGBoost модель или создаёт новую, если не существует"""
    if os.path.exists(model_path):
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return model
    else:
        model = xgb.XGBRegressor(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.07,
            random_state=42
        )
        dummy_X = np.zeros((1, len(feature_columns)))
        dummy_y = [0]
        model.fit(dummy_X, dummy_y)
        model.save_model(model_path)
        return model

def save_xgboost(model, model_path):
    """Сохраняет XGBoost модель"""
    os.makedirs('models', exist_ok=True)
    model.save_model(model_path)


def retrain_random_forest(path_to_data, feature_columns, model_path):
    """
    Полное переобучение RandomForest модели на всех накопленных данных.
    Использует оптимальные гиперпараметры из исследовательского скрипта.
    """
    if not os.path.exists(path_to_data):
        return {"error": f"Файл {path_to_data} не найден"}
    df = pd.read_csv(path_to_data)
    if len(df) == 0:
        return {"error": "Нет данных для обучения"}

    X = df[feature_columns].copy()
    y = df['Result'].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    optimal_params = {
            'n_estimators': 2500,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 0.2,
            'bootstrap': True,
            'max_samples': 0.5,
            'criterion': 'squared_error',
            'ccp_alpha': 0.0001,
            'min_impurity_decrease': 1e-06,
            'min_weight_fraction_leaf': 0.05,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }

    rf_model = RandomForestRegressor(**optimal_params)
    rf_model.fit(X_imputed, y)
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, model_path)
    joblib.dump(imputer, model_path.replace('.pkl', '_imputer.pkl'))
    return {
            "status": "success",
            "n_samples": len(df)
        }

def retrain_dnf_model(data_path, model_path):
    """Переобучение логистической регрессии с весами для учёта важности гонок"""

    df = pd.read_csv(data_path)
    df = df.sort_values(['Year', 'Round']).reset_index(drop=True)
    df['DNF'] = df['Is_finished'].apply(lambda x: 1 if x == 0 else 0)

    dnf_feature_columns = [
        'result_gap_normalized', 'result_vs_team_avg', 'q_x_team', 'avg_pit_x_q',
        'super_ratio_1', 'Average_pit_stop', 'finish_rate_last_5', 'adaptation_score',
        'dnf_rate', 'finish_rate', 'meta_ratio', 'Q', 'temp_humidity_ratio',
        'rank_x_team', 'best_last_3'
    ]
    available_features = [c for c in dnf_feature_columns if c in df.columns]

    X = df[available_features].copy()
    y = df['DNF'].copy()

    n = len(df)
    weights = np.ones(n) * 0.3  # базовый вес для всех

    for i in range(n):
        distance = n - i - 1  # 0 = последняя гонка
        if distance < 5:
            weights[i] = 1.0  # последние 5 гонок
        elif distance < 20:
            weights[i] = 0.6  # предыдущие 15 гонок (5-20)

    weights = weights / weights.sum() * n
    pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(
            penalty='l2',
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    )

    X_imputed = pipeline.named_steps['simpleimputer'].fit_transform(X)
    X_scaled = pipeline.named_steps['standardscaler'].fit_transform(X_imputed)
    pipeline.named_steps['logisticregression'].fit(X_scaled, y, sample_weight=weights)

    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, model_path)
    joblib.dump(available_features, 'models/dnf_features.pkl')
    return pipeline
