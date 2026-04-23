from operator import index

import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
import xgboost as xgb



RF_MODEL_PATH = 'models/randomforest.pkl'
RF_IMPUTER_PATH = 'models/imputer.pkl'
XGB_MODEL_PATH = 'models/xgboost_adaptive.json'


def make_prediction_of_dnf(data, drivers):
    """
    Предсказывает возможные сходы автомобилей с помощью логистической регрессии.
    """
    pipeline = joblib.load(r'models/dnf_model.pkl')
    feature_names = joblib.load(r'models/dnf_features.pkl')
    threshold = 0.55
    X = data[feature_names].copy()
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    probabilities = pipeline.predict_proba(X_imputed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result = {}
    for i, driver in enumerate(drivers):
        result[driver] = int(predictions[i])
    return result


def make_prediction_of_results(features_df, drivers_list):
    """ Комбинированное предсказание: 60% RandomForest + 40% XGBoost """
    rf_model = joblib.load(RF_MODEL_PATH)
    rf_imputer = joblib.load(RF_IMPUTER_PATH)
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGB_MODEL_PATH)

    if hasattr(rf_model, 'feature_names_in_'):
        feature_names = rf_model.feature_names_in_.tolist()
    else:
        feature_names = features_df.columns.tolist()

    for feat in feature_names:
        if feat not in features_df.columns:
            features_df[feat] = 0.0

    X = features_df[feature_names].copy()
    X_imputed = rf_imputer.transform(X)
    X_imputed_df = pd.DataFrame(X_imputed, columns=feature_names, index=X.index)
    pred_rf = rf_model.predict(X_imputed_df)

    X_filled = X.fillna(X.median())
    pred_xgb = xgb_model.predict(X_filled)

    predictions = pred_rf * 0.65  + pred_xgb * 0.35

    results_df = pd.DataFrame({
        'driver': drivers_list,
        'prediction': predictions
    })
    results_df = results_df.sort_values('prediction', ascending=True).reset_index(drop=True)
    results_df['position'] = results_df.index + 1

    return list(zip(results_df['driver'].tolist(), results_df['position'].tolist()))


DRIVER_CODE_TO_NAME = {
    "VER": "Verstappen",
    "HAD": "Hadjar",
    "RUS": "Russell",
    "ANT": "Antonelli",
    "LEC": "Leclerc",
    "HAM": "Hamilton",
    "NOR": "Norris",
    "PIA": "Piastri",
    "ALO": "Alonso",
    "STR": "Stroll",
    "COL": "Colapinto",
    "GAS": "Gasly",
    "ALB": "Albon",
    "SAI": "Sainz",
    "OCO": "Ocon",
    "BEA": "Bearman",
    "LAW": "Lawson",
    "LIN": "Lindblad",
    "HUL": "Hulkenberg",
    "BOR": "Bortoleto",
    "BOT": "Bottas",
    "PER": "Perez"
}

NAME_TO_DRIVER_CODE = {v: k for k, v in DRIVER_CODE_TO_NAME.items()}



def make_prediction_of_results_with_article_analys(features_df, drivers_list, period=10, sentiment_weight=0.15):
    """
    Комбинированное предсказание с автоматическим анализом новостей
    """
    from MotorsportAnalyzer import MotorsportAnalyzer
    from RaceAnalyzer import RaceAnalyzer

    all_scores = {}
    all_counts = {}

    analyzers = [MotorsportAnalyzer(period), RaceAnalyzer(period)]

    for analyzer in analyzers:
        scores = analyzer.get_predictions()
        for driver, score in scores.items():
            all_scores[driver] = all_scores.get(driver, 0) + score
            all_counts[driver] = all_counts.get(driver, 0) + 1

    news_sentiment = {d: all_scores[d] / all_counts[d] for d in all_scores if all_counts[d] > 0}

    rf_model = joblib.load(RF_MODEL_PATH)
    rf_imputer = joblib.load(RF_IMPUTER_PATH)
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGB_MODEL_PATH)

    if hasattr(rf_model, 'feature_names_in_'):
        feature_names = rf_model.feature_names_in_.tolist()
    else:
        feature_names = features_df.columns.tolist()

    for feat in feature_names:
        if feat not in features_df.columns:
            features_df[feat] = 0.0

    X = features_df[feature_names].copy()
    X_imputed = rf_imputer.transform(X)
    X_imputed_df = pd.DataFrame(X_imputed, columns=feature_names, index=X.index)
    pred_rf = rf_model.predict(X_imputed_df)

    X_filled = X.fillna(X.median())
    pred_xgb = xgb_model.predict(X_filled)

    predictions = pred_rf * 0.65 + pred_xgb * 0.35

    results_df = pd.DataFrame({
        'driver_code': drivers_list,
        'driver_name': [DRIVER_CODE_TO_NAME.get(d, d) for d in drivers_list],
        'raw_prediction': predictions
    })

    for i, row in results_df.iterrows():
        driver_name = row['driver_name']
        sentiment = news_sentiment.get(driver_name, 0.0)
        multiplier = 1 + sentiment * sentiment_weight
        results_df.at[i, 'sentiment'] = sentiment
        results_df.at[i, 'prediction'] = row['raw_prediction'] * multiplier

    results_df = results_df.sort_values('prediction', ascending=True).reset_index(drop=True)
    results_df['position'] = results_df.index + 1

    return list(zip(results_df['driver_code'].tolist(), results_df['position'].tolist()))

