import pandas as pd
import json

def load_statistical_data():
    with open(r'f1_statistical_data\drivers_by_teams.json', 'r', encoding='utf-8') as f:
        encoded_drivers = json.load(f)
    with open(r'f1_statistical_data\places.json', 'r', encoding='utf-8') as f:
        places = json.load(f)
    with open(r'f1_statistical_data\drivers_by_teams.json', 'r', encoding='utf-8') as f:
        drivers_by_teams = json.load(f)
    historical_data = pd.read_csv(r'f1_data_by_year\f1_data_pca.csv')
    with open(r'f1_statistical_data\teams_encoded.json', 'r', encoding='utf-8') as f:
        teams_encoded = json.load(f)
    return encoded_drivers, places, drivers_by_teams, historical_data, teams_encoded

def add_features_for_dnf_predictions(data):
    """ Подготавливает признаки для модели логистической регрессии для предсказания сходов гонщиков """
    encoded_drivers, places, drivers_by_teams, historical_data, teams_encoded = load_statistical_data()
    n_drivers = len(data.Driver)
    new_data = pd.DataFrame({
        'Year': [data.Year] * n_drivers,
        'Round': [data.Round] * n_drivers,
        'Driver': data.Driver,
        'Place': [data.Place] * n_drivers,
        'Q': [data.Q.get(d, 20) for d in data.Driver],
        'Start_Position': data.Start_Position,
        'P1': data.P1,
        'P2': data.P2,
        'P3': data.P3,
        'AirTemp': [data.Weather[0]] * n_drivers,
        'TrackTemp': [data.Weather[1]] * n_drivers,
        'Humidity': [data.Weather[2]] * n_drivers,
        'Rainfall': [data.Weather[3]] * n_drivers,
        'WindSpeed': [data.Weather[4]] * n_drivers,
        'Sprint': [data.Sprint.get(d, 0) for d in data.Driver],
    })

    new_data['Driver_encoded'] = new_data['Driver'].map(encoded_drivers)
    new_data['Place_encoded'] = new_data['Place'].map(places)

    driver_to_team = {}
    for team, drivers in drivers_by_teams.items():
        for driver in drivers:
            driver_to_team[driver] = team

    new_data['Team'] = new_data['Driver'].map(driver_to_team)
    new_data['Team_encoded'] = new_data['Team'].map(teams_encoded)

    hist = historical_data.copy()
    driver_avg_all = hist.groupby('Driver_encoded')['Result'].mean()
    team_avg_all = hist.groupby('Team_encoded')['Result'].mean()
    hist['result_vs_team_avg'] = hist['Driver_encoded'].map(driver_avg_all) - hist['Team_encoded'].map(team_avg_all)
    hist['result_vs_team_avg'] = hist['result_vs_team_avg'].fillna(0)

    driver_avg = hist.groupby('Driver_encoded')['Result'].mean()
    team_avg = hist.groupby('Team_encoded')['Result'].mean()
    new_data['result_vs_team_avg'] = new_data['Driver_encoded'].map(driver_avg) - new_data['Team_encoded'].map(team_avg)
    new_data['result_vs_team_avg'] = new_data['result_vs_team_avg'].fillna(0)

    std_val = hist['result_vs_team_avg'].std()
    new_data['result_gap_normalized'] = new_data['result_vs_team_avg'] / (std_val + 1e-6)

    new_data['q_x_team'] = new_data['Q'] * new_data['Team_encoded']

    pitstop_by_place = hist.groupby('Place_encoded')['Average_pit_stop'].median()
    new_data['avg_pit_stop_est'] = new_data['Place_encoded'].map(pitstop_by_place).fillna(
        hist['Average_pit_stop'].median())
    new_data['avg_pit_x_q'] = new_data['avg_pit_stop_est'] * new_data['Q']

    last_points_ratio = hist.groupby('Driver_encoded')['points_ratio_log'].last()
    new_data['points_ratio_log'] = new_data['Driver_encoded'].map(last_points_ratio).fillna(0)
    new_data['super_ratio_1'] = new_data['points_ratio_log'] * new_data['result_gap_normalized']

    team_avg_pitstop = hist.groupby('Team_encoded')['Average_pit_stop'].mean()
    new_data['Average_pit_stop'] = new_data['Team_encoded'].map(team_avg_pitstop)
    new_data['Average_pit_stop'] = new_data['Average_pit_stop'].fillna(hist['Average_pit_stop'].mean())

    def get_driver_rates(driver_enc):
        driver_data = hist[hist['Driver_encoded'] == driver_enc]
        if len(driver_data) > 0:
            finish_rate = driver_data['Is_finished'].mean()
            dnf_rate = 1 - finish_rate
            last_5 = driver_data.sort_values(['Year', 'Round'], ascending=False).head(5)
            finish_rate_last_5 = last_5['Is_finished'].mean() if len(last_5) > 0 else finish_rate
            return finish_rate, dnf_rate, finish_rate_last_5
        return 0.85, 0.15, 0.85

    rates = new_data['Driver_encoded'].apply(get_driver_rates)
    new_data['finish_rate'] = rates.apply(lambda x: x[0])
    new_data['dnf_rate'] = rates.apply(lambda x: x[1])
    new_data['finish_rate_last_5'] = rates.apply(lambda x: x[2])

    new_data['Factorys_or_not'] = new_data['Team'].map(
        lambda x: 1 if x in ['Ferrari', 'Mercedes', 'Red Bull Racing', 'Audi', 'Cadillac'] else 0)
    new_data['relative_money'] = (new_data['Team_encoded'] - new_data['Team_encoded'].mean()) / (new_data['Team_encoded'].std() + 1e-6)

    reg_2022 = hist[hist['Year'] == 2022].groupby('Team_encoded')['result_vs_team_avg'].mean()
    new_data['team_reg_2022_perf'] = new_data['Team_encoded'].map(reg_2022).fillna(0)

    new_data['adaptation_score'] = (new_data['Factorys_or_not'] * 0.3 +
                                    new_data['relative_money'] * 0.4 +
                                    new_data['team_reg_2022_perf'] * 0.3)

    new_data['meta_ratio'] = new_data['points_ratio_log'] / (new_data['result_gap_normalized'].abs() + 1)

    new_data['temp_humidity_ratio'] = new_data['AirTemp'] / (new_data['Humidity'] + 1)

    new_data['rank_x_team'] = new_data['Start_Position'] * new_data['Team_encoded']

    def get_best_last_3(driver_enc):
        driver_data = hist[hist['Driver_encoded'] == driver_enc]
        if len(driver_data) > 0:
            last_3 = driver_data.sort_values(['Year', 'Round'], ascending=False).head(3)
            return last_3['Result'].min()
        return 15.0

    new_data['best_last_3'] = new_data['Driver_encoded'].apply(get_best_last_3)

    feature_columns = [
        'result_gap_normalized', 'result_vs_team_avg', 'q_x_team', 'avg_pit_x_q',
        'super_ratio_1', 'Average_pit_stop', 'finish_rate_last_5', 'adaptation_score',
        'dnf_rate', 'finish_rate', 'meta_ratio', 'Q', 'temp_humidity_ratio',
        'rank_x_team', 'best_last_3'
    ]

    return new_data[feature_columns]