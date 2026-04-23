import pandas as pd
import numpy as np
import json


def load_statistical_data():
    with open(r'f1_statistical_data\engines_manufacters_2026.json', 'r', encoding='utf-8') as f:
        engines = json.load(f)
    with open(r'f1_statistical_data\year_of_birthday_of_drivers.json', 'r', encoding='utf-8') as f:
        age = json.load(f)
    with open(r'f1_statistical_data\team_factory_status.json', 'r', encoding='utf-8') as f:
        factories = json.load(f)
    with open(r'f1_statistical_data\Salary.json', 'r', encoding='utf-8') as f:
        salary = json.load(f)
    with open(r'f1_statistical_data\team_financial_power.json', 'r', encoding='utf-8') as f:
        team_financial_power = json.load(f)
    with open(r'f1_statistical_data\drivers_by_teams.json', 'r', encoding='utf-8') as f:
        drivers_by_teams = json.load(f)
    historical_data = pd.read_csv(r'f1_data_by_year\f1_data_pca.csv')
    with open(r'f1_statistical_data\tracks_speeds.json', 'r', encoding='utf-8') as f:
        tracks_speed = json.load(f)
    with open(r'f1_statistical_data\new_regulations.json', 'r', encoding='utf-8') as f:
        new_regulations = json.load(f)
    with open(r'f1_statistical_data\Speed_of_turns.json', 'r', encoding='utf-8') as f:
        speed_of_turns = json.load(f)
    with open(r'f1_statistical_data\turns.json', 'r', encoding='utf-8') as f:
        turns = json.load(f)
    with open(r'f1_statistical_data\brakes.json', 'r', encoding='utf-8') as f:
        brakes = json.load(f)
    with open(r'f1_statistical_data\tires.json', 'r', encoding='utf-8') as f:
        tires = json.load(f)
    with open(r'f1_statistical_data\drivers.json', 'r', encoding='utf-8') as f:
        encoded_drivers = json.load(f)
    with open(r'f1_statistical_data\places.json', 'r', encoding='utf-8') as f:
        places = json.load(f)
    with open(r'f1_statistical_data\teams_encoded.json', 'r', encoding='utf-8') as f:
        teams_dict = json.load(f)


    return (engines, age, factories, salary,
            team_financial_power, drivers_by_teams, historical_data,
            tracks_speed, new_regulations,speed_of_turns, turns, brakes,
            tires, encoded_drivers, places, teams_dict)

def add_features_for_results_predictions(data):
    """ Подготавливает признаки для модели предсказания результатов гонки """
    (engines, age, factories, salary, team_financial_power, drivers_by_teams,
     historical_data, tracks_speed, new_regulations, speed_of_turns, turns,
     brakes, tires, encoded_drivers, places, team_encoding_dict) = load_statistical_data()
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
    new_data['Team_encoded'] = new_data['Team'].map(team_encoding_dict)

    # points_ratio_log из истории
    last_points_ratio = historical_data.groupby('Driver_encoded')['points_ratio_log'].last()
    new_data['points_ratio_log'] = new_data['Driver_encoded'].map(last_points_ratio).fillna(0)

    # best_last_3 и form_volatility через вспомогательные функции
    best_last_3_dict = calculate_best_last_3(
        historical_data, data.Year, data.Round, new_data['Driver_encoded'].tolist()
    )
    new_data['best_last_3'] = new_data['Driver_encoded'].map(best_last_3_dict)

    form_volatility_dict = calculate_form_volatility(
        historical_data, data.Year, data.Round, new_data['Driver_encoded'].tolist()
    )
    new_data['form_volatility'] = new_data['Driver_encoded'].map(form_volatility_dict)

    # result_vs_team_avg
    driver_avg = historical_data.groupby('Driver_encoded')['Result'].mean()
    team_avg = historical_data.groupby('Team_encoded')['Result'].mean()
    new_data['result_vs_team_avg'] = new_data['Driver_encoded'].map(driver_avg) - new_data['Team_encoded'].map(team_avg)

    # redbull_x_vs_team
    redbull_value = team_encoding_dict.get('Red Bull Racing', 0)
    new_data['redbull_x_vs_team'] = (new_data['Team_encoded'] == redbull_value).astype(int) * new_data[
        'result_vs_team_avg']
    # После строки:
    new_data['result_vs_team_avg'] = new_data['Driver_encoded'].map(driver_avg) - new_data['Team_encoded'].map(team_avg)


    # result_gap_normalized
    std_val = new_data['result_vs_team_avg'].std()
    new_data['result_gap_normalized'] = new_data['result_vs_team_avg'] / (std_val + 1e-6)
    new_data['result_gap_normalized'] = new_data['result_vs_team_avg'] / (std_val + 1e-6)

    # q_x_team
    new_data['q_x_team'] = new_data['Q'] * new_data['Team_encoded']

    # avg_pit_x_q
    pitstop_by_place = historical_data.groupby('Place_encoded')['Average_pit_stop'].median()
    new_data['avg_pit_stop_est'] = new_data['Place_encoded'].map(pitstop_by_place).fillna(
        historical_data['Average_pit_stop'].median())
    new_data['avg_pit_x_q'] = new_data['avg_pit_stop_est'] * new_data['Q']

    # temp_humidity_ratio
    new_data['temp_humidity_ratio'] = new_data['AirTemp'] / (new_data['Humidity'] + 1)

    # Salary и Salary_encoded
    new_data['Salary'] = new_data['Driver'].map(salary)
    salary_mean = historical_data['Salary'].mean()
    salary_std = historical_data['Salary'].std()
    new_data['Salary_encoded'] = (new_data['Salary'] - salary_mean) / salary_std

    # Speed_of_turns
    new_data['Speed_of_turns'] = new_data['Place'].map(speed_of_turns)

    # Regulation_Impact и factory_x_reg
    new_data['Regulation_Impact'] = new_regulations.get(str(data.Year))
    new_data['Factorys_or_not'] = new_data['Team'].map(factories)
    new_data['factory_x_reg'] = new_data['Factorys_or_not'] * new_data['Regulation_Impact']

    # Points_before_race
    points_df = calculate_points_before_race(historical_data, data.Year, data.Round)
    new_data = new_data.merge(points_df, on='Driver_encoded', how='left')
    new_data['Points_before_race'] = new_data['Points_before_race'].fillna(0)

    # Championship_position_before
    new_data['Championship_position_before'] = new_data['Points_before_race'].rank(
        method='dense', ascending=False
    ).astype(int)

    # Quantity_of_turns
    new_data['Quantity_of_turns'] = new_data['Place'].map(turns)

    # tracktemp_x_points_log
    new_data['tracktemp_x_points_log'] = new_data['TrackTemp'] * np.log1p(new_data['Points_before_race'])

    # rank_x_team
    new_data['rank_x_team'] = new_data['Championship_position_before'] * new_data['Team_encoded']

    # Speed_of_track, Brake_load и производные
    new_data['Speed_of_track'] = new_data['Place'].map(tracks_speed)
    new_data['Brake_load'] = new_data['Place'].map(brakes)
    new_data['brake_x_speed_track'] = new_data['Brake_load'] * new_data['Speed_of_track']
    new_data['brake_x_wind'] = new_data['Brake_load'] * new_data['WindSpeed']

    # Average_pit_stop по командам
    team_avg_pitstop = historical_data.groupby('Team_encoded')['Average_pit_stop'].mean()
    new_data['Average_pit_stop'] = new_data['Team_encoded'].map(team_avg_pitstop)
    overall_avg_pitstop = historical_data['Average_pit_stop'].mean()
    new_data['Average_pit_stop'] = new_data['Average_pit_stop'].fillna(overall_avg_pitstop)

    # meta_ratio
    new_data['meta_ratio'] = new_data['points_ratio_log'] / (new_data['result_gap_normalized'].abs() + 1)

    # avg_last_5
    avg_last_5_dict = calculate_avg_last_5(historical_data, data.Year, data.Round, new_data['Driver_encoded'].tolist())
    new_data['avg_last_5'] = new_data['Driver_encoded'].map(avg_last_5_dict)

    # Result_last_year
    mask = (historical_data['Year'] == data.Year - 1) & (historical_data['Round'] == data.Round)
    last_year_result = historical_data.loc[mask].set_index('Driver_encoded')['Result']
    new_data['Result_last_year'] = new_data['Driver_encoded'].map(last_year_result).fillna(20)

    # avg_season
    avg_season_dict = calculate_avg_season_accurate(
        historical_data, data.Year, data.Round, new_data['Driver_encoded'].tolist(), new_data['Place_encoded'].iloc[0]
    )
    new_data['avg_season'] = new_data['Driver_encoded'].map(avg_season_dict)

    # super_ratio_1
    new_data['super_ratio_1'] = new_data['points_ratio_log'] * new_data['result_gap_normalized']

    # Tire_wear и form_x_difficulty
    new_data['Tire_wear'] = new_data['Place'].map(tires)
    track_difficulty = (new_data['Tire_wear'] + new_data['Speed_of_turns'] + new_data['Brake_load']) / 3
    # avg_points_ratio должен быть вычислен до использования
    avg_points_ratio_dict = calculate_avg_points_ratio(historical_data, data.Year, data.Round,
                                                       new_data['Driver_encoded'].tolist())
    new_data['avg_points_ratio'] = new_data['Driver_encoded'].map(avg_points_ratio_dict)
    new_data['form_x_difficulty'] = new_data['avg_points_ratio'] * track_difficulty

    # temp_gap и temp_gap_x_team
    new_data['temp_gap'] = new_data['TrackTemp'] - new_data['AirTemp']
    new_data['temp_gap_x_team'] = new_data['temp_gap'] * new_data['Team_encoded']

    # Team_financial_power и relative_money (нормализация по текущему году)
    new_data['Team_financial_power'] = new_data['Team'].map(team_financial_power)
    mean_fp = new_data['Team_financial_power'].mean()
    std_fp = new_data['Team_financial_power'].std()
    new_data['relative_money'] = (new_data['Team_financial_power'] - mean_fp) / std_fp if std_fp > 0 else 0

    team_avg_2022 = historical_data[historical_data['Year'] == 2022].groupby('Team_encoded')['Result'].mean()
    driver_avg_2022 = historical_data[historical_data['Year'] == 2022].groupby('Driver_encoded')['Result'].mean()

    # Создаем словарь team_encoded -> средний результат пилота
    team_driver_avg = {}
    for team, drivers in drivers_by_teams.items():
        team_enc = team_encoding_dict.get(team)
        if team_enc is None:
            continue
        driver_encs = [encoded_drivers.get(d) for d in drivers if encoded_drivers.get(d) is not None]
        driver_results = [driver_avg_2022.get(d) for d in driver_encs if driver_avg_2022.get(d) is not None]
        if driver_results:
            team_driver_avg[team_enc] = np.mean(driver_results)

    # Вычисляем result_vs_team_avg для 2022 года
    reg_2022_values = {}
    for driver_enc in driver_avg_2022.index:
        driver_result = driver_avg_2022[driver_enc]
        team_enc = None
        for team, drivers in drivers_by_teams.items():
            for driver in drivers:
                if encoded_drivers.get(driver) == driver_enc:
                    team_enc = team_encoding_dict.get(team)
                    break
            if team_enc is not None:
                break
        if team_enc is not None and team_enc in team_avg_2022.index:
            team_result = team_avg_2022[team_enc]
            reg_2022_values[team_enc] = driver_result - team_result

    # Создаем Series для team_reg_2022_perf
    reg_2022_series = pd.Series(reg_2022_values)
    new_data['team_reg_2022_perf'] = new_data['Team_encoded'].map(reg_2022_series).fillna(0)

    # adaptation_score
    new_data['adaptation_score'] = (new_data['Factorys_or_not'] * 0.3 +
                                    new_data['relative_money'] * 0.4 +
                                    new_data['team_reg_2022_perf'] * 0.3)

    # consistency_vs_avg
    new_data['consistency_vs_avg'] = new_data['form_volatility'] / (new_data['avg_last_5'] + 1)

    # points_to_leader
    leader_points = new_data['Points_before_race'].max()
    new_data['points_to_leader'] = leader_points - new_data['Points_before_race']

    # tracktemp_x_salary
    new_data['tracktemp_x_salary'] = new_data['TrackTemp'] * new_data['Salary_encoded']

    # brake_x_turns
    new_data['brake_x_turns'] = new_data['Brake_load'] * new_data['Quantity_of_turns']

    # pitstop_to_speed_ratio
    new_data['pitstop_to_speed_ratio'] = new_data['Average_pit_stop'] / (new_data['Speed_of_track'] + 1)

    # tracktemp_x_start
    new_data['tracktemp_x_start'] = new_data['TrackTemp'] * new_data['Start_Position']

    new_data['Engine'] = new_data['Team'].map(engines)
    new_data['Engine_Ferrari'] = (new_data['Engine'] == 'Ferrari').astype(int)

    final_columns = [
        'result_vs_team_avg',
        'redbull_x_vs_team',
        'result_gap_normalized',
        'q_x_team',
        'avg_pit_x_q',
        'Start_Position',
        'temp_humidity_ratio',
        'Salary_encoded',
        'Speed_of_turns',
        'factory_x_reg',
        'Points_before_race',
        'Quantity_of_turns',
        'Engine_Ferrari',
        'tracktemp_x_points_log',
        'rank_x_team',
        'brake_x_speed_track',
        'brake_x_wind',
        'Average_pit_stop',
        'meta_ratio',
        'avg_last_5',
        'Result_last_year',
        'avg_season',
        'super_ratio_1',
        'form_x_difficulty',
        'temp_gap_x_team',
        'adaptation_score',
        'consistency_vs_avg',
        'points_to_leader',
        'tracktemp_x_salary',
        'brake_x_turns',
        'Q',
        'best_last_3',
        'form_volatility',
        'pitstop_to_speed_ratio',
        'avg_points_ratio',
        'tracktemp_x_start', 'Year',
        'Round',
        'Driver_encoded',
        'points_ratio_log',
        'Place_encoded',
        'Team_encoded',
        'Salary'
    ]
    return new_data[final_columns]


def add_engines(data, engines):
    """Добавляем признаки производителей двигателей"""
    data['Engine'] = data['Team_encoded'].map(engines)
    data['Engine_Ferrari'] = data['Engine'].apply(lambda x: 1 if x == 'Ferrari' else 0)
    data['Engine_Mercedes'] = data['Engine'].apply(lambda x: 1 if x == 'Mercedes' else 0)
    del data['Engine']
    return data

def normalize_salary(data, database):
    """Нормализуем зарплату"""
    salary_mean = database['Salary'].mean()
    salary_std = database['Salary'].std()
    data['Salary_encoded'] = data['Salary'].apply(lambda x: (x - salary_mean) / salary_std)

    return data


def calculate_points_before_race(historical_data, current_year, current_round):
    """
    Рассчитывает Points_before_race для каждого пилота перед текущей гонкой
    """
    race_points = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }
    sprint_points = {
        1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1
    }

    if current_round == 1:
        drivers_in_season = historical_data[historical_data['Year'] == current_year]['Driver_encoded'].unique()
        points_df = pd.DataFrame({
            'Driver_encoded': drivers_in_season,
            'Points_before_race': 0
        })
        return points_df

    prev_round = current_round - 1

    prev_race_data = historical_data[
        (historical_data['Year'] == current_year) &
        (historical_data['Round'] == prev_round)
        ].copy()

    if prev_race_data.empty:
        drivers_in_season = historical_data[historical_data['Year'] == current_year]['Driver_encoded'].unique()
        points_df = pd.DataFrame({
            'Driver_encoded': drivers_in_season,
            'Points_before_race': 0
        })
        return points_df

    prev_points = prev_race_data[['Driver_encoded', 'Points_before_race']].drop_duplicates('Driver_encoded')
    prev_points_dict = dict(zip(prev_points['Driver_encoded'], prev_points['Points_before_race']))
    current_points = []

    for _, race in prev_race_data.iterrows():
        driver = race['Driver_encoded']
        base_points = prev_points_dict.get(driver, 0)
        result = race['Result']
        if result in race_points:
            base_points += race_points[result]
        if 'Sprint' in race and not pd.isna(race['Sprint']) and race['Sprint'] > 0:
            sprint_pos = race['Sprint']
            if sprint_pos in sprint_points:
                base_points += sprint_points[sprint_pos]
        current_points.append({'Driver_encoded': driver, 'Points_before_race': base_points})
    points_df = pd.DataFrame(current_points).drop_duplicates('Driver_encoded')
    return points_df


def add_points_to_new_race(new_race_df, historical_data):
    """
    Добавляет колонку Points_before_race в DataFrame новой гонки
    """

    current_year = new_race_df['Year'].iloc[0]
    current_round = new_race_df['Round'].iloc[0]
    points_df = calculate_points_before_race(historical_data, current_year, current_round)
    result_df = new_race_df.merge(points_df, on='Driver_encoded', how='left')
    result_df['Points_before_race'] = result_df['Points_before_race'].fillna(0)
    return result_df


def calculate_avg_last_5(historical_data, current_year, current_round, drivers_list):
    """ Рассчитывает средний результат за последние 5 гонок для каждого пилота """
    past_races = historical_data[
        ((historical_data['Year'] < current_year) |
         ((historical_data['Year'] == current_year) &
          (historical_data['Round'] < current_round)))
    ].copy()

    avg_last_5_dict = {}

    for driver in drivers_list:
        # Берем все гонки пилота из прошлого
        driver_races = past_races[past_races['Driver_encoded'] == driver].sort_values(
            ['Year', 'Round'], ascending=False)
        last_5 = driver_races.head(5)

        if len(last_5) > 0:
            avg_last_5 = last_5['Result'].mean()
        else:
            avg_last_5 = past_races['Result'].mean()
        avg_last_5_dict[driver] = avg_last_5

    return avg_last_5_dict


def calculate_avg_season_accurate(historical_data, current_year, current_round, drivers_list, place):
    """ Расчёт avg_season"""
    current_season = historical_data[
        (historical_data['Year'] == current_year) &
        (historical_data['Round'] < current_round)
        ]

    last_year_data = historical_data[
        (historical_data['Year'] == current_year - 1) &
        (historical_data['Place_encoded'] == place)
        ][['Driver_encoded', 'Result']].rename(columns={'Result': 'Result_last_year'})

    avg_season_dict = {}

    for driver in drivers_list:
        driver_races = current_season[current_season['Driver_encoded'] == driver].sort_values('Round')
        season_results = driver_races['Result'].tolist()
        last_year_res = last_year_data[last_year_data['Driver_encoded'] == driver]['Result_last_year']
        if not last_year_res.empty:
            prev_year_res = last_year_res.iloc[0]
            if prev_year_res > 20:
                prev_year_res = 20.0
        else:
            prev_year_res = None
        if len(season_results) > 0:
            avg_season = sum(season_results) / len(season_results)
        elif prev_year_res is not None:
            avg_season = prev_year_res
        else:
            avg_season = historical_data['Result'].mean()
        avg_season_dict[driver] = avg_season
    return avg_season_dict


def calculate_avg_points_ratio(historical_data, current_year, current_round, drivers_list):
    """ Рассчитывает среднее отношение очков пилота к очкам напарника """
    if current_round == 1:
        return {driver: 1.0 for driver in drivers_list}

    past_races = historical_data[
        (historical_data['Year'] == current_year) &
        (historical_data['Round'] < current_round)
        ]

    avg_ratio_dict = {}

    for driver in drivers_list:
        driver_races = past_races[past_races['Driver_encoded'] == driver]

        if len(driver_races) > 0 and 'points_ratio' in driver_races.columns:
            avg_ratio = driver_races['points_ratio'].mean()
        else:
            avg_ratio = 1.0
        avg_ratio_dict[driver] = avg_ratio
    return avg_ratio_dict


def calculate_best_last_3(historical_data, current_year, current_round, drivers_list):
    """ Рассчитывает лучший результат за последние 3 гонки для каждого пилота (best_last_3) """
    past_data = historical_data[
        ((historical_data['Year'] < current_year) |
         ((historical_data['Year'] == current_year) &
          (historical_data['Round'] < current_round)))
    ].copy()

    best_last_3_dict = {}

    for driver in drivers_list:
        driver_races = past_data[past_data['Driver_encoded'] == driver].sort_values(
            ['Year', 'Round'], ascending=False
        )
        last_3 = driver_races.head(3)

        if len(last_3) > 0:
            best_last_3 = last_3['Result'].min()
        else:
            best_last_3 = 15.0
        best_last_3_dict[driver] = best_last_3

    return best_last_3_dict


def calculate_form_volatility(historical_data, current_year, current_round, drivers_list):
    """ Рассчитывает form_volatility за последние 5 гонок """
    past_data = historical_data[
        ((historical_data['Year'] < current_year) |
         ((historical_data['Year'] == current_year) &
          (historical_data['Round'] < current_round)))
    ].copy()

    form_volatility_dict = {}

    for driver in drivers_list:
        driver_races = past_data[past_data['Driver_encoded'] == driver].sort_values(
            ['Year', 'Round'], ascending=False)
        last_5 = driver_races.head(5)
        if len(last_5) >= 2:
            form_volatility = last_5['Result'].std()
        elif len(last_5) == 1:
            form_volatility = 0
        else:
            form_volatility = past_data.groupby('Driver_encoded')['Result'].std().mean()
            if pd.isna(form_volatility):
                form_volatility = 3.0

        form_volatility_dict[driver] = form_volatility
    return form_volatility_dict


def calculate_result_vs_team_avg_current_season(historical_data, current_year, current_round, drivers_list):
    """
    Рассчитывает result_vs_team_avg на основе данных текущего сезона
    """
    current_season = historical_data[
        (historical_data['Year'] == current_year) &
        (historical_data['Round'] < current_round)
        ]

    if len(current_season) == 0:
        return {driver: 0 for driver in drivers_list}
    driver_avg_current = current_season.groupby('Driver_encoded')['Result'].mean()
    team_avg_current = current_season.groupby('Team_encoded')['Result'].mean()

    result = {}
    for driver_enc in drivers_list:
        d_avg = driver_avg_current.get(driver_enc, 0)
        t_avg = team_avg_current.get(driver_enc, 0)
        result[driver_enc] = d_avg - t_avg

    return result