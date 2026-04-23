import uvicorn
from fastapi import FastAPI, HTTPException
import pandas as pd
import os
from MotorsportAnalyzer import MotorsportAnalyzer
from RaceAnalyzer import RaceAnalyzer
from pandas import DataFrame
from pydantic import BaseModel
from typing import List, Dict
import xgboost as xgb
import numpy as np

from add_features_for_results_predictions import add_features_for_results_predictions
from add_features_for_dnf_predictions import add_features_for_dnf_predictions
from predictions import make_prediction_of_dnf, make_prediction_of_results
from predictions import make_prediction_of_results_with_article_analys
from relearning_models import retrain_random_forest, save_xgboost, load_or_create_xgboost, retrain_dnf_model

app = FastAPI()

DATA_FILE = 'f1_data_by_year/f1_data_pca.csv'
RF_MODEL_PATH = 'models/randomforest.pkl'
DNF_MODEL_PATH = 'models/dnf_model.pkl'
RF_IMPUTER_PATH = 'models/imputer.pkl'
XGB_MODEL_PATH = 'models/xgboost_adaptive.json'

FEATURE_COLUMNS_RES = [
    'result_vs_team_avg', 'redbull_x_vs_team', 'result_gap_normalized',
    'q_x_team', 'avg_pit_x_q', 'Start_Position', 'temp_humidity_ratio',
    'Salary_encoded', 'Speed_of_turns', 'factory_x_reg', 'Points_before_race',
    'Quantity_of_turns', 'Engine_Ferrari', 'tracktemp_x_points_log', 'rank_x_team',
    'brake_x_speed_track', 'brake_x_wind', 'Average_pit_stop', 'meta_ratio',
    'avg_last_5', 'Result_last_year', 'avg_season', 'super_ratio_1',
    'form_x_difficulty', 'temp_gap_x_team', 'adaptation_score', 'consistency_vs_avg',
    'points_to_leader', 'tracktemp_x_salary', 'brake_x_turns', 'Q', 'best_last_3',
    'form_volatility', 'pitstop_to_speed_ratio', 'avg_points_ratio', 'tracktemp_x_start'
]

FEATURE_COLUMNS_DNF = [
    'result_gap_normalized', 'result_vs_team_avg', 'q_x_team', 'avg_pit_x_q',
    'super_ratio_1', 'Average_pit_stop', 'finish_rate_last_5', 'adaptation_score',
    'dnf_rate', 'finish_rate', 'meta_ratio', 'Q', 'temp_humidity_ratio',
    'rank_x_team', 'best_last_3'
]

class PredictionInput(BaseModel):
    """Данные для предсказаний"""
    Year: int
    Round: int
    Place: str
    Driver: List[str]
    P1: List[int]
    P2: List[int]
    P3: List[int]
    Q: Dict[str, int]
    Start_Position: List[int]
    Sprint: Dict[str, int]
    Weather: List[float]


class NewDataInput(BaseModel):
    """Данные для занесения результатов гонки"""
    Year: int
    Round: int
    Place: str
    Driver: List[str]
    Result: List[int]
    P1: List[int]
    P2: List[int]
    P3: List[int]
    Q: Dict[str, int]
    Start_Position: List[int]
    Sprint: Dict[str, int]
    Weather: List[float]


@app.post("/predict")
def predict(data: PredictionInput):
    """Метод для сбора и оформления предсказаний"""
    drivers = data.Driver
    data_to_prediction_dnf = add_features_for_dnf_predictions(data)
    dnf_results = make_prediction_of_dnf(data_to_prediction_dnf, drivers)
    features = add_features_for_results_predictions(data)
    results = make_prediction_of_results_with_article_analys(features[FEATURE_COLUMNS_RES], drivers)

    pred_positions = {driver: pos for driver, pos in results}
    temp_results = []
    for driver in drivers:
        temp_results.append({
            'driver': driver,
            'position': pred_positions[driver],
            'dnf': dnf_results[driver]
        })

    finished = [r for r in temp_results if r['dnf'] == 0]
    dnf_list = [r for r in temp_results if r['dnf'] == 1]
    finished_sorted = sorted(finished, key=lambda x: x['position'])
    dnf_sorted = sorted(dnf_list, key=lambda x: x['position'])
    all_results = finished_sorted + dnf_sorted

    merge_results = []
    for idx, result in enumerate(all_results, 1):
        merge_results.append((result['driver'], idx, result['dnf']))

    return merge_results


@app.post("/update_data")
def update_data(data: NewDataInput):
    """Добавляет новую гонку в историю и дообучает XGBoost модель и логистическую регрессию"""
    try:
        new_data = add_features_for_results_predictions(data)
        new_data['Result'] = data.Result
        new_data['Is_finished'] = [1 if pos > 22 else 0 for pos in data.Result]
        if os.path.exists(DATA_FILE):
            existing_data = pd.read_csv(DATA_FILE)
            year = data.Year
            round = data.Round
            mask = ((existing_data['Year'] == year) & (existing_data['Round'] == round))
            if mask.any():
                existing_data = existing_data[~mask]
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            updated_data = new_data
        updated_data.to_csv(DATA_FILE, index=False)

        X_new = new_data[FEATURE_COLUMNS_RES].copy()
        y_new = new_data['Result'].copy()
        X_new = X_new.replace([np.inf, -np.inf], np.nan)
        xgb_model = load_or_create_xgboost(XGB_MODEL_PATH, FEATURE_COLUMNS_RES)
        X_filled = X_new.fillna(X_new.median())
        temp_model = xgb.XGBRegressor(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.07,
            random_state=42
        )
        temp_model.fit(X_filled, y_new, xgb_model=xgb_model)
        save_xgboost(temp_model, XGB_MODEL_PATH)
        retrain_dnf_model(DATA_FILE, DNF_MODEL_PATH)
        return {
            "status": "success",
            "message": f"Гонка {data.Round} добавлена. XGBoost дообучен. DNF модель переобучена. Всего записей: {len(updated_data)}"
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/retrain_rf")
def start_retrain_random_forest():
    """Переобучение модели Random Forest"""
    try:
        result = retrain_random_forest(DATA_FILE, FEATURE_COLUMNS_RES, RF_MODEL_PATH)
        if "error" in result:
            return {"status": "error", "message": result["error"]}
        return {
            "status": "success",
            "message": f"Модель RandomForest успешно переобучена на {result.get('n_samples', 0)} записях"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/delete_features")
def delete_useless_features():
    """Вспомогательный метод для удаления неиспользуемых в
    модели признаков, которые есть в собранных данных файла f1_data_pca.csv
    """
    try:
        file_path = 'f1_data_by_year/f1_data_pca.csv'
        data = pd.read_csv('f1_data_by_year/f1_data_pca.csv')
        basic_columns = ['Year', 'Round', 'Driver_encoded', 'points_ratio_log',
                         'Result', 'Place_encoded', 'Team_encoded', 'Salary', 'Is_finished']
        feats = FEATURE_COLUMNS_RES + basic_columns

        existing_cols = [c for c in feats if c in data.columns]
        data = data[existing_cols]
        data.to_csv(file_path, index=False)
        return {
            "status": "success",
            "message": f"Обработано. Осталось {len(data)} строк, {len(data.columns)} колонок",
            "rows": len(data),
            "columns": len(data.columns)
        }

    except Exception as e:
        print(f"Ошибка: {e}")
        return {"error": str(e)}

@app.post("/delete_last_22_result")
def delete_last_22_result():
    data = pd.read_csv(DATA_FILE)
    data = data.iloc[:-22]
    data.to_csv(DATA_FILE, index=False)

if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=5000)




