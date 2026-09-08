
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from datetime import datetime, date

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Predicción de precios de vuelos",
    page_icon="✈️",
    layout="wide"
)

APP_DIR = Path(__file__).parent
FECHA_CADUCIDAD = date.fromisoformat("2026-08-04")

hoy = datetime.today().date()

if hoy > FECHA_CADUCIDAD:
    st.error("Este enlace temporal ha caducado.")
    st.info("La app estaba configurada para funcionar hasta el " + FECHA_CADUCIDAD.strftime("%d/%m/%Y") + ".")
    st.stop()

dias_restantes = (FECHA_CADUCIDAD - hoy).days

# ============================================================
# CARGA DE MODELO Y DATOS
# ============================================================

@st.cache_resource
def cargar_modelo():
    return joblib.load(APP_DIR / "modelo_top99.joblib")

@st.cache_data
def cargar_datos():
    with open(APP_DIR / "features_top99.json", "r", encoding="utf-8") as f:
        features = json.load(f)

    defaults = pd.read_csv(APP_DIR / "defaults_features_top99.csv")
    route_defaults = pd.read_csv(APP_DIR / "route_defaults_top99.csv")
    route_market_defaults = pd.read_csv(APP_DIR / "route_market_defaults_top99.csv")
    route_options = pd.read_csv(APP_DIR / "route_options_top99.csv")
    route_market_options = pd.read_csv(APP_DIR / "route_market_options_top99.csv")

    return features, defaults, route_defaults, route_market_defaults, route_options, route_market_options

modelo = cargar_modelo()
features, defaults_df, route_defaults, route_market_defaults, route_options, route_market_options = cargar_datos()

defaults = dict(zip(defaults_df["feature"], defaults_df["value"]))

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def quarter_from_month(m):
    return int((int(m) - 1) // 3 + 1)

def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def set_feature(feature_values, possible_names, value):
    for name in possible_names:
        if name in feature_values:
            feature_values[name] = value

def get_route_defaults(route_key):
    if route_key is None:
        return {}

    fila = route_defaults[route_defaults["route_key"].astype(str) == str(route_key)]

    if len(fila) == 0:
        return {}

    fila = fila.iloc[0].to_dict()
    fila.pop("route_key", None)

    return {
        k: safe_float(v, defaults.get(k, 0.0))
        for k, v in fila.items()
        if k in features
    }

def get_route_market_defaults(route_market_key):
    if route_market_key is None:
        return {}

    fila = route_market_defaults[
        route_market_defaults["route_market_key"].astype(str) == str(route_market_key)
    ]

    if len(fila) == 0:
        return {}

    fila = fila.iloc[0].to_dict()
    fila.pop("route_market_key", None)

    return {
        k: safe_float(v, defaults.get(k, 0.0))
        for k, v in fila.items()
        if k in features
    }

def construir_vector_base(route_key, route_market_key):
    feature_values = {
        f: safe_float(defaults.get(f, 0.0), 0.0)
        for f in features
    }

    # Primero defaults globales, luego mercado, luego ruta concreta
    feature_values.update(get_route_market_defaults(route_market_key))
    feature_values.update(get_route_defaults(route_key))

    return feature_values

def predecir(feature_values):
    X_pred = pd.DataFrame([[feature_values[f] for f in features]], columns=features)
    X_pred = X_pred.replace([np.inf, -np.inf], np.nan)
    pred = modelo.predict(X_pred)[0]
    return float(pred), X_pred

def crear_label_ruta(row):
    return (
        str(row["route_key"]) +
        " | " +
        str(row["startingAirport"]) +
        " → " +
        str(row["destinationAirport"]) +
        " | mercado: " +
        str(row["route_market_key"]) +
        " | N=" +
        str(int(row["N_train"]))
    )

# ============================================================
# CABECERA
# ============================================================

st.title("✈️ Predicción interactiva de precios de vuelos")
st.caption("Modelo Top 99% de importancia - Fase 4")

st.info(
    "URL temporal activa hasta el **" +
    FECHA_CADUCIDAD.strftime("%d/%m/%Y") +
    "**. Quedan **" +
    str(dias_restantes) +
    " días** si la sesión que sirve la app sigue activa."
)

with st.sidebar:
    st.header("Modelo")
    st.write("Modelo: **HistGradientBoostingRegressor**")
    st.write("Features:", len(features))
    st.write("Set de entrenamiento: notebook TFM")
    st.write("Caducidad:", FECHA_CADUCIDAD.strftime("%d/%m/%Y"))

# ============================================================
# SELECCIÓN DE RUTA
# ============================================================

st.header("1. Selección de ruta")

if route_options.empty:
    st.error("No hay rutas disponibles en route_options_top99.csv.")
    st.stop()

route_options = route_options.copy()
route_options["label"] = route_options.apply(crear_label_ruta, axis=1)

col_a, col_b = st.columns([2, 1])

with col_a:
    ruta_label = st.selectbox(
        "Ruta aeropuerto-aeropuerto disponible en el modelo",
        route_options["label"].tolist()
    )

selected_route = route_options[route_options["label"] == ruta_label].iloc[0]

route_key = str(selected_route["route_key"])
route_market_key = str(selected_route["route_market_key"])
starting_airport = str(selected_route["startingAirport"])
destination_airport = str(selected_route["destinationAirport"])
origin_market = str(selected_route["origin_market"])
dest_market = str(selected_route["dest_market"])

with col_b:
    st.metric("Precio medio train de la ruta", f"{safe_float(selected_route['precio_medio_train']):.2f} $")
    st.metric("Observaciones train", f"{int(selected_route['N_train']):,}")

st.write(
    "**Ruta seleccionada:**",
    route_key,
    "| **Mercado:**",
    route_market_key
)

# ============================================================
# INPUTS BÁSICOS
# ============================================================

st.header("2. Características del vuelo")

feature_values = construir_vector_base(route_key, route_market_key)

col1, col2, col3 = st.columns(3)

with col1:
    search_date = st.date_input(
        "Fecha de búsqueda",
        value=date(2022, 7, 1)
    )

with col2:
    flight_date = st.date_input(
        "Fecha de vuelo",
        value=date(2022, 8, 1)
    )

with col3:
    days_to_departure = max((flight_date - search_date).days, 0)

    st.metric(
        "Días hasta salida",
        days_to_departure
    )

search_month = search_date.month
search_quarter = quarter_from_month(search_month)
search_dayofweek = search_date.weekday()

flight_month = flight_date.month
flight_quarter = quarter_from_month(flight_month)
flight_dayofweek = flight_date.weekday()

is_weekend_departure = 1 if flight_dayofweek >= 5 else 0
is_summer = 1 if flight_month in [6, 7, 8] else 0
is_peak_month = 1 if flight_month in [6, 7, 8, 11, 12] else 0
is_thanksgiving_period = 1 if (flight_month == 11 and 20 <= flight_date.day <= 30) else 0
is_christmas_period = 1 if (flight_month == 12 and 15 <= flight_date.day <= 31) else 0

set_feature(feature_values, ["days_to_departure"], days_to_departure)
set_feature(feature_values, ["search_month"], search_month)
set_feature(feature_values, ["search_quarter"], search_quarter)
set_feature(feature_values, ["search_dayofweek"], search_dayofweek)
set_feature(feature_values, ["flight_month"], flight_month)
set_feature(feature_values, ["flight_quarter"], flight_quarter)
set_feature(feature_values, ["flight_dayofweek"], flight_dayofweek)
set_feature(feature_values, ["is_weekend_departure"], is_weekend_departure)
set_feature(feature_values, ["is_summer"], is_summer)
set_feature(feature_values, ["is_peak_month"], is_peak_month)
set_feature(feature_values, ["is_thanksgiving_period"], is_thanksgiving_period)
set_feature(feature_values, ["is_christmas_period"], is_christmas_period)

col4, col5, col6 = st.columns(3)

with col4:
    seats_remaining = st.slider(
        "Asientos restantes",
        min_value=0,
        max_value=10,
        value=int(round(safe_float(feature_values.get("seatsRemaining", 5), 5)))
    )

with col5:
    travel_duration_mins = st.number_input(
        "Duración del viaje en minutos",
        min_value=30,
        max_value=2000,
        value=int(round(safe_float(feature_values.get("travelDuration_mins", 300), 300))),
        step=15
    )

with col6:
    total_distance = st.number_input(
        "Distancia total del viaje",
        min_value=0,
        max_value=6000,
        value=int(round(safe_float(feature_values.get("totalTravelDistance", 1000), 1000))),
        step=50
    )

set_feature(feature_values, ["seatsRemaining"], seats_remaining)
set_feature(feature_values, ["travelDuration_mins"], travel_duration_mins)
set_feature(feature_values, ["totalTravelDistance"], total_distance)

col7, col8, col9, col10, col11 = st.columns(5)

with col7:
    is_non_stop = st.selectbox("Vuelo directo", ["No", "Sí"], index=1)

with col8:
    is_basic = st.selectbox("Basic Economy", ["No", "Sí"], index=0)

with col9:
    is_refundable = st.selectbox("Reembolsable", ["No", "Sí"], index=0)

with col10:
    is_multi_airline = st.selectbox("Varias aerolíneas", ["No", "Sí"], index=0)

with col11:
    has_business = st.selectbox("Segmento business/first", ["No", "Sí"], index=0)

is_non_stop_num = 1 if is_non_stop == "Sí" else 0
is_basic_num = 1 if is_basic == "Sí" else 0
is_refundable_num = 1 if is_refundable == "Sí" else 0
is_multi_airline_num = 1 if is_multi_airline == "Sí" else 0
has_business_num = 1 if has_business == "Sí" else 0

set_feature(feature_values, ["isNonStop_num", "isNonStop"], is_non_stop_num)
set_feature(feature_values, ["isBasicEconomy_num", "isBasicEconomy"], is_basic_num)
set_feature(feature_values, ["isRefundable_num", "isRefundable"], is_refundable_num)
set_feature(feature_values, ["is_multi_airline"], is_multi_airline_num)
set_feature(feature_values, ["has_business_segment"], has_business_num)

# elapsedDays suele ser 0 en Expedia, pero si está en features se deja editable
if "elapsedDays" in feature_values:
    feature_values["elapsedDays"] = st.number_input(
        "elapsedDays",
        min_value=0,
        max_value=5,
        value=int(round(safe_float(feature_values.get("elapsedDays", 0), 0))),
        step=1
    )

# ============================================================
# VARIABLES AVANZADAS
# ============================================================

st.header("3. Variables avanzadas del modelo")

st.markdown(
    "Las variables no introducidas manualmente se rellenan automáticamente con valores medios de la ruta, del mercado o del conjunto de entrenamiento. "
    "Puedes modificarlas si quieres simular otro escenario."
)

with st.expander("Editar todas las variables usadas por el modelo"):
    df_features_editor = pd.DataFrame({
        "feature": features,
        "value": [safe_float(feature_values.get(f, 0.0), 0.0) for f in features]
    })

    df_features_editor = st.data_editor(
        df_features_editor,
        disabled=["feature"],
        use_container_width=True,
        height=420
    )

    for _, row in df_features_editor.iterrows():
        feature_values[str(row["feature"])] = safe_float(row["value"], feature_values.get(str(row["feature"]), 0.0))

# ============================================================
# PREDICCIÓN
# ============================================================

st.header("4. Predicción")

precio_predicho, X_pred = predecir(feature_values)

col_pred1, col_pred2, col_pred3 = st.columns(3)

with col_pred1:
    st.metric("Precio predicho del vuelo", f"{precio_predicho:.2f} $")

with col_pred2:
    precio_medio_ruta = safe_float(selected_route["precio_medio_train"])
    diferencia = precio_predicho - precio_medio_ruta
    st.metric("Diferencia vs media ruta", f"{diferencia:.2f} $")

with col_pred3:
    if precio_medio_ruta != 0:
        diferencia_pct = diferencia / precio_medio_ruta * 100
        st.metric("Diferencia porcentual", f"{diferencia_pct:.2f}%")
    else:
        st.metric("Diferencia porcentual", "N/A")

st.success(
    "Predicción realizada para " +
    starting_airport +
    " → " +
    destination_airport +
    " (" +
    route_market_key +
    ")"
)

# ============================================================
# DETALLE DEL VECTOR
# ============================================================

with st.expander("Ver vector final enviado al modelo"):
    st.dataframe(X_pred.T.rename(columns={0: "valor"}), use_container_width=True)

st.divider()

st.caption(
    "App temporal · Modelo Top 99% Fase 4 · Los valores no introducidos manualmente se completan con medias de ruta/mercado/entrenamiento."
)
