import streamlit as st
import pandas as pd
import requests

# Настройка страницы
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide"
)

# Кастомный CSS для красивого оформления
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .main-header {
        background: linear-gradient(90deg, #e10600 0%, #ff4d4d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .mode-selector {
        background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .weather-card {
        background: linear-gradient(135deg, #1a2634 0%, #2c3e50 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        color: white;
    }
    .weather-card .stNumberInput input, .weather-card .stSelectbox div {
        background-color: #34495e !important;
        color: white !important;
        border: 2px solid #e10600 !important;
        border-radius: 5px !important;
        font-weight: bold !important;
    }
    .weather-card .stNumberInput input:hover, .weather-card .stSelectbox div:hover {
        border-color: #ff4d4d !important;
        box-shadow: 0 0 8px #e10600 !important;
    }
    .weather-card .stNumberInput input:disabled {
        background-color: #2c3e50 !important;
        border-color: #666 !important;
        color: #aaa !important;
    }
    .weather-card .st-bb {
        background-color: rgba(255,255,255,0.15);
        border-radius: 5px;
        padding: 5px 10px;
    }
    .weather-card .stNumberInput, .weather-card .stSelectbox {
        background-color: rgba(255,255,255,0.1);
        border-radius: 5px;
        padding: 2px 5px;
    }
    .weather-card .st-b7 {
        color: #ffd700 !important;
        font-weight: bold;
    }
    .weather-card .st-cx {
        background-color: rgba(255,255,255,0.2);
    }
    .weather-card hr {
        border-color: rgba(255,255,255,0.2);
    }
    .weather-card .stMarkdown {
        color: white !important;
    }
    .weather-card label {
        color: #ffd700 !important;
        font-weight: bold;
    }
    .driver-table {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .podium-1 {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
        color: #000;
        font-weight: bold;
    }
    .podium-2 {
        background: linear-gradient(135deg, #c0c0c0 0%, #e0e0e0 100%);
        color: #000;
        font-weight: bold;
    }
    .podium-3 {
        background: linear-gradient(135deg, #cd7f32 0%, #ed9e5e 100%);
        color: #000;
        font-weight: bold;
    }
    .driver-input-card {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 4px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        background: white;
    }
    .driver-input-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .driver-code {
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 5px;
    }
    .driver-team {
        font-size: 0.8em;
        color: #6c757d;
        margin-bottom: 8px;
    }
    /* Цвета команд */
    .team-redbull { border-left-color: #1E41FF; }
    .team-redbull .driver-code { color: #1E41FF; }
    .team-ferrari { border-left-color: #DC0000; }
    .team-ferrari .driver-code { color: #DC0000; }
    .team-mclaren { border-left-color: #FF8700; }
    .team-mclaren .driver-code { color: #FF8700; }
    .team-mercedes { border-left-color: #00D2BE; }
    .team-mercedes .driver-code { color: #00D2BE; }
    .team-astonmartin { border-left-color: #006F62; }
    .team-astonmartin .driver-code { color: #006F62; }
    .team-alpine { border-left-color: #0090FF; }
    .team-alpine .driver-code { color: #0090FF; }
    .team-haas { border-left-color: #B6BABD; }
    .team-haas .driver-code { color: #B6BABD; }
    .team-williams { border-left-color: #005AFF; }
    .team-williams .driver-code { color: #005AFF; }
    .team-audi { border-left-color: #E63946; }
    .team-audi .driver-code { color: #E63946; }
    .team-rb { border-left-color: #2B4562; }
    .team-rb .driver-code { color: #2B4562; }
    .team-cadillac { border-left-color: #000000; }
    .team-cadillac .driver-code { color: #000000; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
        background-color: #f8f9fa;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e10600 !important;
        color: white !important;
    }
    .validation-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #ffc107;
    }
    .validation-success {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
    .retrain-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Функция для получения класса команды
def get_team_class(team):
    team_classes = {
        "McLaren": "team-mclaren",
        "Mercedes": "team-mercedes",
        "Red Bull": "team-redbull",
        "Ferrari": "team-ferrari",
        "Williams": "team-williams",
        "Racing Bulls": "team-rb",
        "Aston Martin": "team-astonmartin",
        "Haas": "team-haas",
        "Audi": "team-audi",
        "Alpine": "team-alpine",
        "Cadillac": "team-cadillac"
    }
    return team_classes.get(team, "")


# Заголовок
st.markdown("""
<div class="main-header">
    <h1 style="margin:0">🏎️ F1 Race Predictor</h1>
    <p style="margin:0; opacity:0.9;">Предсказание и анализ результатов гонок</p>
</div>
""", unsafe_allow_html=True)

# Выбор режима работы
st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
mode = st.radio(
    "Выберите режим работы:",
    ["🔮 Предсказать результаты", "📊 Загрузить результаты гонки", "🔄 Перезагрузка модели"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# Инициализация session state
if 'df_data' not in st.session_state:
    # АКТУАЛЬНЫЙ СПИСОК ПИЛОТОВ 2026
    drivers = [
        # McLaren
        "NOR", "PIA",
        # Mercedes
        "RUS", "ANT",
        # Red Bull
        "VER", "HAD",
        # Ferrari
        "LEC", "HAM",
        # Williams
        "ALB", "SAI",
        # Racing Bulls
        "LAW", "LIN",
        # Aston Martin
        "ALO", "STR",
        # Haas
        "OCO", "BEA",
        # Audi
        "HUL", "BOR",
        # Alpine
        "GAS", "COL",
        # Cadillac
        "BOT", "PER"
    ]

    # АКТУАЛЬНЫЕ КОМАНДЫ 2026
    teams = {
        "NOR": "McLaren", "PIA": "McLaren",
        "RUS": "Mercedes", "ANT": "Mercedes",
        "VER": "Red Bull", "HAD": "Red Bull",
        "LEC": "Ferrari", "HAM": "Ferrari",
        "ALB": "Williams", "SAI": "Williams",
        "LAW": "Racing Bulls", "LIN": "Racing Bulls",
        "ALO": "Aston Martin", "STR": "Aston Martin",
        "OCO": "Haas", "BEA": "Haas",
        "HUL": "Audi", "BOR": "Audi",
        "GAS": "Alpine", "COL": "Alpine",
        "BOT": "Cadillac", "PER": "Cadillac"
    }

    # Создаём DataFrame с начальными значениями
    data = []
    for driver in drivers:
        data.append({
            "Пилот": driver,
            "Команда": teams.get(driver, ""),
            "P1": 10,
            "P2": 10,
            "P3": 10,
            "Q": 5,
            "Старт": 5,
            "Спринт": 0,
            "Результат": 10,
            "is_dnf": False
        })

    st.session_state.df_data = pd.DataFrame(data)
    st.session_state.data_version = 0

# Режим перезагрузки модели
if mode == "🔄 Перезагрузка модели":
    st.markdown('<div class="retrain-card">', unsafe_allow_html=True)
    st.markdown("### 🔄 Перезагрузка модели Random Forest")
    st.markdown("Нажмите кнопку ниже для переобучения модели на актуальных данных")

    retrain_url = st.text_input("🔌 API URL (retrain)", value="http://localhost:5000/retrain_rf", key="api_url_retrain")

    if st.button("🚀 ЗАПУСТИТЬ ПЕРЕОБУЧЕНИЕ", type="primary", width="stretch"):
        with st.spinner("🔄 Переобучение модели..."):
            try:
                response = requests.post(retrain_url, timeout=60)

                if response.status_code == 200:
                    st.success("✅ Модель успешно переобучена!")
                    try:
                        st.json(response.json())
                    except:
                        st.text(response.text)
                else:
                    st.error(f"❌ Ошибка: {response.status_code}")
                    st.text(response.text)
            except requests.exceptions.ConnectionError:
                st.error("❌ Не удалось подключиться к API. Убедитесь, что сервер запущен.")
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Основной контент для режимов предсказания и загрузки
col1, col2 = st.columns([1, 2])

with col1:
    # Карточка с погодой
    st.markdown('<div class="weather-card">', unsafe_allow_html=True)
    st.markdown("### 🌤️ Погода на гонку")

    year = st.number_input("📅 Год", min_value=2024, max_value=2026, value=2026, step=1, key="year_input")
    round_num = st.number_input("🔢 Раунд", min_value=1, max_value=24, value=1, step=1, key="round_input")

    place = st.selectbox(
        "📍 Трасса",
        ["Bahrain", "Jeddah", "Melbourne", "Suzuka", "Shanghai",
         "Miami", "Imola", "Monaco", "Barcelona", "Montreal",
         "Spielberg", "Silverstone", "Budapest", "Spa", "Zandvoort",
         "Monza", "Baku", "Singapore", "Austin", "Mexico City",
         "Interlagos", "Las Vegas", "Losail", "Yas Marina"],
        key="place_select"
    )

    st.markdown("---")

    # Погодные условия
    weather_temp = st.slider("🌡️ Температура воздуха (°C)", 0.0, 50.0, 25.0, 0.5, key="temp")
    weather_track = st.slider("🔥 Температура трассы (°C)", 0.0, 70.0, 35.0, 0.5, key="track")
    weather_humidity = st.slider("💧 Влажность (%)", 0.0, 100.0, 60.0, 1.0, key="humidity")
    weather_rain = st.selectbox("☔ Дождь", ["Нет", "Легкий", "Сильный"], key="rain")
    weather_wind = st.slider("💨 Ветер (м/с)", 0.0, 20.0, 2.5, 0.1, key="wind")

    rain_map = {"Нет": 0, "Легкий": 1, "Сильный": 2}
    rainfall = rain_map[weather_rain]

    # Разные URL для разных режимов
    if "Предсказать" in mode:
        api_url = st.text_input("🔌 API URL (predict)", value="http://localhost:5000/predict", key="api_url_predict")
        button_text = "🔮 ПРЕДСКАЗАТЬ РЕЗУЛЬТАТЫ"
    else:
        api_url = st.text_input("🔌 API URL (update)", value="http://localhost:5000/update_data", key="api_url_update")
        button_text = "📊 ЗАГРУЗИТЬ РЕЗУЛЬТАТЫ"

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Таблица с пилотами
    st.markdown('<div class="driver-table">', unsafe_allow_html=True)

    # Заголовок с информацией о режиме
    if "Предсказать" in mode:
        st.markdown("#### ✏️ Введите данные для предсказания")
    else:
        st.markdown("#### ✏️ Введите результаты гонки")
        st.markdown("⚠️ **Важно:** Позиции в гонке должны быть уникальными (1-22). Используйте кнопку DNF для схода")

    # Создаем вкладки в зависимости от режима
    if "Предсказать" in mode:
        # В режиме предсказания только 6 вкладок (без гонки)
        tab_p1, tab_p2, tab_p3, tab_q, tab_start, tab_sprint = st.tabs(
            ["🟢 P1", "🟡 P2", "🔵 P3", "🏁 Квалификация", "🚦 Старт", "⚡ Спринт"]
        )
        show_race_tab = False
    else:
        # В режиме загрузки все 7 вкладок
        tab_p1, tab_p2, tab_p3, tab_q, tab_start, tab_sprint, tab_race = st.tabs(
            ["🟢 P1", "🟡 P2", "🔵 P3", "🏁 Квалификация", "🚦 Старт", "⚡ Спринт", "🏁 Гонка"]
        )
        show_race_tab = True

    with tab_p1:
        st.markdown("##### P1 - Первая практика (позиции 1-22)")
        cols = st.columns(2)
        for idx, (i, row) in enumerate(st.session_state.df_data.iterrows()):
            col_idx = idx % 2
            with cols[col_idx]:
                team_class = get_team_class(row['Команда'])
                st.markdown(f"""
                <div class="driver-input-card {team_class}">
                    <div class="driver-code">{row['Пилот']}</div>
                    <div class="driver-team">{row['Команда']}</div>
                </div>
                """, unsafe_allow_html=True)

                input_key = f"p1_{row['Пилот']}_{st.session_state.data_version}"
                new_value = st.number_input(
                    "Позиция",
                    min_value=1,
                    max_value=22,
                    value=int(row['P1']),
                    step=1,
                    key=input_key,
                    label_visibility="collapsed"
                )
                if new_value != row['P1']:
                    st.session_state.df_data.at[i, 'P1'] = new_value

    with tab_p2:
        st.markdown("##### P2 - Вторая практика (позиции 1-22)")
        cols = st.columns(2)
        for idx, (i, row) in enumerate(st.session_state.df_data.iterrows()):
            col_idx = idx % 2
            with cols[col_idx]:
                team_class = get_team_class(row['Команда'])
                st.markdown(f"""
                <div class="driver-input-card {team_class}">
                    <div class="driver-code">{row['Пилот']}</div>
                    <div class="driver-team">{row['Команда']}</div>
                </div>
                """, unsafe_allow_html=True)

                input_key = f"p2_{row['Пилот']}_{st.session_state.data_version}"
                new_value = st.number_input(
                    "Позиция",
                    min_value=1,
                    max_value=22,
                    value=int(row['P2']),
                    step=1,
                    key=input_key,
                    label_visibility="collapsed"
                )
                if new_value != row['P2']:
                    st.session_state.df_data.at[i, 'P2'] = new_value

    with tab_p3:
        st.markdown("##### P3 - Третья практика (позиции 1-22)")
        cols = st.columns(2)
        for idx, (i, row) in enumerate(st.session_state.df_data.iterrows()):
            col_idx = idx % 2
            with cols[col_idx]:
                team_class = get_team_class(row['Команда'])
                st.markdown(f"""
                <div class="driver-input-card {team_class}">
                    <div class="driver-code">{row['Пилот']}</div>
                    <div class="driver-team">{row['Команда']}</div>
                </div>
                """, unsafe_allow_html=True)

                input_key = f"p3_{row['Пилот']}_{st.session_state.data_version}"
                new_value = st.number_input(
                    "Позиция",
                    min_value=1,
                    max_value=22,
                    value=int(row['P3']),
                    step=1,
                    key=input_key,
                    label_visibility="collapsed"
                )
                if new_value != row['P3']:
                    st.session_state.df_data.at[i, 'P3'] = new_value

    with tab_q:
        st.markdown("##### Квалификация (Q) - позиции 1-22")
        cols = st.columns(2)
        for idx, (i, row) in enumerate(st.session_state.df_data.iterrows()):
            col_idx = idx % 2
            with cols[col_idx]:
                team_class = get_team_class(row['Команда'])
                st.markdown(f"""
                <div class="driver-input-card {team_class}">
                    <div class="driver-code">{row['Пилот']}</div>
                    <div class="driver-team">{row['Команда']}</div>
                </div>
                """, unsafe_allow_html=True)

                input_key = f"q_{row['Пилот']}_{st.session_state.data_version}"
                new_value = st.number_input(
                    "Позиция",
                    min_value=1,
                    max_value=22,
                    value=int(row['Q']),
                    step=1,
                    key=input_key,
                    label_visibility="collapsed"
                )
                if new_value != row['Q']:
                    st.session_state.df_data.at[i, 'Q'] = new_value

    with tab_start:
        st.markdown("##### Старт - позиции 1-22")
        cols = st.columns(2)
        for idx, (i, row) in enumerate(st.session_state.df_data.iterrows()):
            col_idx = idx % 2
            with cols[col_idx]:
                team_class = get_team_class(row['Команда'])
                st.markdown(f"""
                <div class="driver-input-card {team_class}">
                    <div class="driver-code">{row['Пилот']}</div>
                    <div class="driver-team">{row['Команда']}</div>
                </div>
                """, unsafe_allow_html=True)

                input_key = f"start_{row['Пилот']}_{st.session_state.data_version}"
                new_value = st.number_input(
                    "Позиция",
                    min_value=1,
                    max_value=22,
                    value=int(row['Старт']),
                    step=1,
                    key=input_key,
                    label_visibility="collapsed"
                )
                if new_value != row['Старт']:
                    st.session_state.df_data.at[i, 'Старт'] = new_value

    with tab_sprint:
        st.markdown("##### ⚡ Спринт (0 если не проводится или DNF, 1-22 позиции)")
        cols = st.columns(2)
        for idx, (i, row) in enumerate(st.session_state.df_data.iterrows()):
            col_idx = idx % 2
            with cols[col_idx]:
                team_class = get_team_class(row['Команда'])
                st.markdown(f"""
                <div class="driver-input-card {team_class}">
                    <div class="driver-code">{row['Пилот']}</div>
                    <div class="driver-team">{row['Команда']}</div>
                </div>
                """, unsafe_allow_html=True)

                input_key = f"sprint_{row['Пилот']}_{st.session_state.data_version}"
                new_value = st.number_input(
                    "Позиция",
                    min_value=0,
                    max_value=22,
                    value=int(row['Спринт']),
                    step=1,
                    key=input_key,
                    label_visibility="collapsed"
                )
                if new_value != row['Спринт']:
                    st.session_state.df_data.at[i, 'Спринт'] = new_value

    # Вкладка гонки - только в режиме загрузки
    if show_race_tab:
        with tab_race:
            st.markdown("##### 🏁 Результаты гонки (позиции 1-22, все значения должны быть уникальными)")

            st.markdown("---")

            # Проверяем текущие результаты на дубликаты (исключая DNF)
            results = []
            for i, row in st.session_state.df_data.iterrows():
                if row['is_dnf']:
                    results.append(30)
                else:
                    results.append(row['Результат'])

            results_no_dnf = [r for r in results if r != 30]
            unique_results = set(results_no_dnf)

            if len(unique_results) != len(results_no_dnf):
                st.markdown(f"""
                <div class="validation-warning">
                    ⚠️ Обнаружены дублирующиеся позиции! Найдено дубликатов: {len(results_no_dnf) - len(unique_results)}
                </div>
                """, unsafe_allow_html=True)

                # Показываем дубликаты в компактном виде
                duplicate_values = []
                for result in set([r for r in results_no_dnf if results_no_dnf.count(r) > 1]):
                    drivers_with_duplicate = []
                    for i, row in st.session_state.df_data.iterrows():
                        if not row['is_dnf'] and row['Результат'] == result:
                            drivers_with_duplicate.append(row['Пилот'])
                    duplicate_values.append(f"Позиция {result}: {', '.join(drivers_with_duplicate)}")

                if duplicate_values:
                    st.markdown("**Дублирующиеся позиции:**")
                    for dup in duplicate_values:
                        st.markdown(f"- {dup}")
            else:
                st.markdown(f"""
                <div class="validation-success">
                    ✅ Все позиции уникальны
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Ввод результатов для каждого пилота
            cols = st.columns(2)
            for idx, (i, row) in enumerate(st.session_state.df_data.iterrows()):
                col_idx = idx % 2
                with cols[col_idx]:
                    team_class = get_team_class(row['Команда'])

                    # Добавляем индикатор DNF
                    dnf_indicator = " ⚡ DNF" if row['is_dnf'] else ""

                    st.markdown(f"""
                    <div class="driver-input-card {team_class}">
                        <div class="driver-code">{row['Пилот']}{dnf_indicator}</div>
                        <div class="driver-team">{row['Команда']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Создаем два столбца для поля ввода и кнопки DNF
                    col_num, col_dnf = st.columns([3, 1])

                    with col_num:
                        if row['is_dnf']:
                            # Если DNF, показываем текст вместо поля ввода
                            st.markdown("**🚫 DNF**", help="Сход с дистанции")
                        else:
                            input_key = f"result_{row['Пилот']}_{st.session_state.data_version}"
                            new_value = st.number_input(
                                "Позиция в гонке",
                                min_value=1,
                                max_value=22,
                                value=int(row['Результат']),
                                step=1,
                                key=input_key,
                                label_visibility="collapsed",
                                help="Введите позицию (1-22)"
                            )
                            if new_value != row['Результат']:
                                st.session_state.df_data.at[i, 'Результат'] = new_value

                    with col_dnf:
                        button_label = "✅ Финиш" if row['is_dnf'] else "💀 DNF"

                        if st.button(button_label, key=f"dnf_{row['Пилот']}_{i}_{st.session_state.data_version}",
                                     width="stretch"):
                            # Переключаем состояние DNF
                            new_dnf_state = not row['is_dnf']
                            st.session_state.df_data.at[i, 'is_dnf'] = new_dnf_state
                            if new_dnf_state:
                                st.session_state.df_data.at[i, 'Результат'] = 10
                            else:
                                st.session_state.df_data.at[i, 'Результат'] = 10
                            st.rerun()

    # Кнопка для сброса
    col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
    with col_reset2:
        if st.button("🔄 Сбросить все значения по умолчанию", width="stretch"):
            for i in range(len(st.session_state.df_data)):
                st.session_state.df_data.at[i, 'P1'] = 10
                st.session_state.df_data.at[i, 'P2'] = 10
                st.session_state.df_data.at[i, 'P3'] = 10
                st.session_state.df_data.at[i, 'Q'] = 5
                st.session_state.df_data.at[i, 'Старт'] = 5
                st.session_state.df_data.at[i, 'Спринт'] = 0
                st.session_state.df_data.at[i, 'Результат'] = 10
                st.session_state.df_data.at[i, 'is_dnf'] = False
            st.session_state.data_version += 1
            st.rerun()

    st.caption("💡 Все изменения сохраняются автоматически. Для спринта: 0 = не проводится или DNF")
    st.markdown('</div>', unsafe_allow_html=True)

# Кнопка действия
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    action_button = st.button(
        button_text,
        type="primary",
        width="stretch",
        key="action_button"
    )

# Обработка действия
if action_button:
    # Базовая структура данных
    base_data = {
        "Year": year,
        "Round": round_num,
        "Place": place,
        "Driver": st.session_state.df_data["Пилот"].tolist(),
        "P1": st.session_state.df_data["P1"].astype(int).tolist(),
        "P2": st.session_state.df_data["P2"].astype(int).tolist(),
        "P3": st.session_state.df_data["P3"].astype(int).tolist(),
        "Q": dict(zip(st.session_state.df_data["Пилот"], st.session_state.df_data["Q"].astype(int))),
        "Start_Position": st.session_state.df_data["Старт"].astype(int).tolist(),
        "Sprint": dict(
            zip(st.session_state.df_data["Пилот"], st.session_state.df_data["Спринт"].astype(int))
        ),
        "Weather": [weather_temp, weather_track, weather_humidity, rainfall, weather_wind]
    }

    # Добавляем результаты для режима загрузки
    if "Загрузить" in mode:
        # Формируем результаты с учетом DNF (30)
        results = []
        for i, row in st.session_state.df_data.iterrows():
            if row['is_dnf']:
                results.append(30)
            else:
                results.append(row['Результат'])

        results_no_dnf = [r for r in results if r != 30]

        if len(set(results_no_dnf)) != len(results_no_dnf):
            st.error("❌ Обнаружены дублирующиеся позиции в результатах гонки! Пожалуйста, исправьте.")

            # Показываем дубликаты
            duplicate_rows = []
            for r in set([r for r in results_no_dnf if results_no_dnf.count(r) > 1]):
                for i, row in st.session_state.df_data.iterrows():
                    if not row['is_dnf'] and row['Результат'] == r:
                        duplicate_rows.append({"Пилот": row['Пилот'], "Команда": row['Команда'], "Результат": r})

            if duplicate_rows:
                st.dataframe(pd.DataFrame(duplicate_rows), width="stretch")

            st.stop()

        base_data["Result"] = results

    # Показываем что отправляем
    with st.expander("📤 Отправляемые данные"):
        st.json(base_data)

    # Отправка запроса
    with st.spinner("🔄 Отправка данных..."):
        try:
            response = requests.post(api_url, json=base_data, timeout=30)

            if response.status_code == 200:
                try:
                    result = response.json()
                except requests.exceptions.JSONDecodeError:
                    st.error("❌ Сервер вернул невалидный JSON")
                    st.text(response.text)
                    st.stop()

                if "Предсказать" in mode:
                    st.success("✅ Предсказание успешно получено!")

                    if isinstance(result, list) and len(result) > 0:
                        first_item = result[0]
                        if isinstance(first_item, list) and len(first_item) >= 3:
                            finished = [item for item in result if item[2] == 0]
                            dnf = [item for item in result if item[2] == 1]

                            finished_sorted = sorted(finished, key=lambda x: x[1])

                            st.markdown("### 🏆 Результаты предсказания")

                            col_res1, col_res2 = st.columns(2)

                            with col_res1:
                                st.markdown("#### 🏆 Подиум")

                                podium_styles = ["podium-1", "podium-2", "podium-3"]
                                podium_medals = ["🥇", "🥈", "🥉"]

                                for i, (driver, pos, dnf_flag) in enumerate(finished_sorted[:3], 1):
                                    team = dict(zip(st.session_state.df_data["Пилот"],
                                                    st.session_state.df_data["Команда"])).get(driver, "")
                                    st.markdown(
                                        f'<div class="prediction-card {podium_styles[i - 1]}">'
                                        f'{podium_medals[i - 1]} {driver} ({team}): <b>{pos:.0f}</b>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )

                            with col_res2:
                                st.markdown("#### 📊 Статистика")
                                pred_positions = [item[1] for item in finished_sorted]
                                dnf_count = len(dnf)

                                if pred_positions:
                                    st.metric("Средняя позиция", f"{sum(pred_positions) / len(pred_positions):.2f}")
                                    st.metric("Лучшая позиция", f"{min(pred_positions):.0f}")
                                    st.metric("Худшая позиция", f"{max(pred_positions):.0f}")

                                st.metric("Прогнозируемые сходы", dnf_count)

                            st.markdown("#### 📋 Полная таблица результатов")

                            results_data = []
                            teams_dict = dict(zip(st.session_state.df_data["Пилот"],
                                                  st.session_state.df_data["Команда"]))

                            for driver, pos, dnf_flag in finished_sorted:
                                results_data.append({
                                    "Место": f"{pos:.0f}",
                                    "Пилот": driver,
                                    "Команда": teams_dict.get(driver, ""),
                                    "Статус": "🏁 Финиш"
                                })

                            for driver, pos, dnf_flag in sorted(dnf, key=lambda x: x[0]):
                                results_data.append({
                                    "Место": "DNF",
                                    "Пилот": driver,
                                    "Команда": teams_dict.get(driver, ""),
                                    "Статус": "⚡ Сход"
                                })

                            results_df = pd.DataFrame(results_data)

                            def highlight_status(val):
                                if val == "🏁 Финиш":
                                    return 'background-color: #90EE90'
                                elif val == "⚡ Сход":
                                    return 'background-color: #FFB6C1'
                                return ''

                            st.dataframe(
                                results_df.style.map(highlight_status, subset=['Статус']),
                                width="stretch",
                                hide_index=True
                            )

                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Скачать результаты (CSV)",
                                csv,
                                f"f1_predictions_{place}_{year}.csv",
                                "text/csv",
                                width="stretch"
                            )

                        else:
                            st.warning("Неожиданный формат ответа от API")
                            st.json(result)
                    elif isinstance(result, dict):
                        st.json(result)
                    else:
                        st.warning("Неожиданный формат ответа от API")
                        st.write(result)

                else:
                    st.success("✅ Данные успешно загружены!")
                    if isinstance(result, dict):
                        st.json(result)
                    else:
                        st.write(result)

            elif response.status_code == 422:
                st.error("❌ Ошибка валидации данных. Проверьте формат.")
                try:
                    st.json(response.json())
                except:
                    st.text(response.text)
            else:
                st.error(f"❌ Ошибка API: {response.status_code}")
                st.text(response.text)

        except requests.exceptions.ConnectionError:
            st.error("❌ Не удалось подключиться к API. Убедитесь, что сервер запущен.")
            st.info("💡 Запустите сервер командой: uvicorn app_api:app --reload")
        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")

# Футер
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        <p>🏎️ F1 Race Predictor</p>
        <p style='font-size: 0.8em;'>Данные за 2018-2025 | Предсказания на 2026</p>
    </div>
    """,
    unsafe_allow_html=True
)