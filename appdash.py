import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from catboost import CatBoostClassifier

# ── Dropdown CSS fix (DARKLY puts white text on white background) ────────────
DROPDOWN_CSS = """
body .Select-value-label,
body .Select-input > input         { color: #212529 !important; }
body .Select-placeholder           { color: #6c757d !important; }
body .Select-menu-outer,
body .Select-option                { background:#fff !important; color:#212529 !important; }
body .Select-option.is-focused     { background:#e9ecef !important; }
body .Select-option.is-selected    { background:#0d6efd !important; color:#fff !important; }
body .dash-dropdown .Select__single-value,
body .dash-dropdown .Select__input-container { color:#212529 !important; }
body .dash-dropdown .Select__placeholder     { color:#6c757d !important; }
body .dash-dropdown .Select__menu            { background:#fff !important; }
body .dash-dropdown .Select__option          { color:#212529 !important; }
body .dash-dropdown .Select__option--is-focused   { background:#e9ecef !important; }
body .dash-dropdown .Select__option--is-selected  { background:#0d6efd !important; color:#fff !important; }
"""

# ── App init ────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Predicción de Retrasos Aéreos",
)
server = app.server


_orig_index = app.index_string
app.index_string = _orig_index.replace(
    "</head>", f"<style>{DROPDOWN_CSS}</style></head>"
)

def cargar_modelo():
    modelo = CatBoostClassifier()
    modelo.load_model("models/modelotarea1.cbm")
    return modelo

def cargar_datos():
    return pd.read_csv("data/vuelos_clima_1778363496.csv")

df = cargar_datos()
modelo = cargar_modelo()

FEATURES = [
    "dep_iata", "arr_iata", "ruta", "weather_temp", "weather_temp_max",
    "weather_temp_min", "delta_temp", "weather_pressure", "weather_humidity",
    "weather_visibility", "weather_wind_speed", "weather_rain_1h",
    "weather_clouds", "hora", "dia_semana", "hora_pico",
    "vuelo_nocturno", "clima_severo", "humedad_alta",
]

DIAS = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sáb", "Dom"]

def metric_card(label, value):
    return dbc.Card(
        dbc.CardBody([
            html.P(label, className="text-muted mb-1", style={"fontSize": "0.8rem"}),
            html.H4(value, className="mb-0 text-white fw-bold"),
        ]),
        className="bg-dark border-secondary text-center shadow",
    )

retrasos_apt = (
    df[df["delayed"] > 0]
    .groupby("dep_iata")["delayed"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)
retrasos_apt.columns = ["Aeropuerto", "Retraso Promedio (min)"]

fig_airports = px.bar(
    retrasos_apt,
    x="Aeropuerto",
    y="Retraso Promedio (min)",
    color="Retraso Promedio (min)",
    color_continuous_scale="Viridis",
    title="Top 15 Aeropuertos con Mayor Retraso Promedio",
    template="plotly_dark",
)
fig_airports.update_layout(
    coloraxis_showscale=False,
    margin=dict(t=50, b=20, l=20, r=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

tab_datos = dbc.Tab(
    label="Datos",
    tab_id="tab-datos",
    children=[
        dbc.Row([
            dbc.Col(metric_card("Promedio de Retraso", f"{df['delayed'].mean():.2f} min"), width=6),
            dbc.Col(metric_card("Retraso Máximo", f"{df['delayed'].max():.0f} min"), width=6),
        ], className="mb-4 mt-3"),
        html.Hr(className="border-secondary"),
        dcc.Graph(figure=fig_airports, config={"displayModeBar": False}),
    ],
)


# ── Tab 2 – Rendimiento ──────────────────────────────────────────────────────
importancias = modelo.get_feature_importance()
feat_imp = (
    pd.DataFrame({"Feature": FEATURES, "Importance": importancias})
    .sort_values("Importance", ascending=True)
    .tail(10)
)

fig_imp = px.bar(
    feat_imp,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="Viridis",
    title="Top 10 Variables más Influyentes",
    template="plotly_dark",
)
fig_imp.update_layout(
    coloraxis_showscale=False,
    margin=dict(t=50, b=20, l=20, r=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis_title="",
)

tab_rendimiento = dbc.Tab(
    label="Rendimiento del Modelo",
    tab_id="tab-rendimiento",
    children=[
        dbc.Row([
            dbc.Col(metric_card("Accuracy",  "99.85%"), width=3),
            dbc.Col(metric_card("Precision", "99.21%"), width=3),
            dbc.Col(metric_card("Recall",    "97.35%"), width=3),
            dbc.Col(metric_card("F1-Score",  "0.98"),   width=3),
        ], className="mb-4 mt-3"),
        html.Hr(className="border-secondary"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_imp, config={"displayModeBar": False}), width=7),
            dbc.Col([
                dbc.Alert([
                    html.Strong("¿Qué significan estos datos?"),
                    html.Ul([
                        html.Li("Variables Categóricas: La ruta y el aeropuerto de origen son los predictores más fuertes."),
                        html.Li("Clima: La visibilidad y velocidad del viento impactan en el umbral de seguridad."),
                        html.Li("Temporalidad: Las 'Horas Pico' ayudan al modelo a identificar saturación de pista."),
                    ], className="mt-2 mb-0"),
                ], color="info"),
                dbc.Alert("Umbral de Decisión: 0.30", color="warning", className="mt-3"),
            ], width=5, className="d-flex flex-column justify-content-center"),
        ]),
    ],
)


# ── Tab 3 – Predicción ───────────────────────────────────────────────────────
tab_prediccion = dbc.Tab(
    label="Predicción",
    tab_id="tab-prediccion",
    children=[
        html.H5("Simulador de Estado de Vuelo", className="mt-3 mb-1"),
        html.P("Ingresa los detalles para predecir la probabilidad de retraso en tiempo real.",
               className="text-muted mb-4"),
        dbc.Row([
            dbc.Col([
                html.P("Detalles del Vuelo", className="fw-bold mb-3"),
                dbc.Label("Aeropuerto de Origen"),
                dcc.Dropdown(
                    id="origen",
                    options=[{"label": v, "value": v} for v in sorted(df["dep_iata"].unique())],
                    value=df["dep_iata"].iloc[0],
                    className="mb-3",
                    style={"color": "#212529"},
                    optionHeight=35,
                ),
                dbc.Label("Aeropuerto de Destino"),
                dcc.Dropdown(
                    id="destino",
                    options=[{"label": v, "value": v} for v in sorted(df["arr_iata"].unique())],
                    value=df["arr_iata"].iloc[0],
                    className="mb-3",
                    style={"color": "#212529"},
                    optionHeight=35,
                ),
                dbc.Label("Día de la semana"),
                dcc.Slider(
                    id="dia",
                    min=0, max=6, step=1, value=0,
                    marks={i: d for i, d in enumerate(DIAS)},
                    className="mb-4",
                ),
                dbc.Label("Hora del vuelo (0–23)"),
                dcc.Slider(
                    id="hora-vuelo",
                    min=0, max=23, step=1, value=12,
                    marks={h: str(h) for h in range(0, 24, 3)},
                    className="mb-3",
                ),
            ], width=6),


            dbc.Col([
                html.P("Condiciones Climáticas", className="fw-bold mb-3"),
                dbc.Label("Temperatura (°C)"),
                dbc.Input(id="temp", type="number", value=20, className="mb-3"),
                dbc.Label("Humedad (%)"),
                dcc.Slider(
                    id="humedad",
                    min=0, max=100, step=1, value=50,
                    marks={v: f"{v}%" for v in range(0, 101, 25)},
                    className="mb-4",
                ),
                dbc.Label("Velocidad del Viento (km/h)"),
                dbc.Input(id="viento", type="number", value=10, className="mb-3"),
                dbc.Label("Visibilidad (metros)"),
                dbc.Input(id="visibilidad", type="number", value=10000, className="mb-3"),
            ], width=6),
        ]),

        html.Hr(className="border-secondary"),

        dbc.Button(
            "Calcular Probabilidad de Retraso",
            id="btn-predecir",
            color="success",
            size="lg",
            className="w-100 mb-4",
        ),

        # Result placeholder
        html.Div(id="resultado-prediccion"),
    ],
)


# ── Sidebar ──────────────────────────────────────────────────────────────────
sidebar = dbc.Card(
    dbc.CardBody([
        html.H6("ℹ️ Información del Dataset", className="text-white fw-bold mb-3"),
        html.P(f"Vuelos analizados: {len(df):,}", className="text-muted mb-1"),
        html.P(f"Aeropuertos únicos: {df['dep_iata'].nunique()}", className="text-muted mb-1"),
        html.Hr(className="border-secondary my-3"),
        html.Small("Actualización automática cada 5 min", className="text-secondary"),
    ]),
    className="bg-dark border-secondary shadow h-100",
)

app.layout = dbc.Container(
    fluid=True,
    className="py-4 px-4",
    children=[
        dcc.Interval(id="refresh-interval", interval=5 * 60 * 1000, n_intervals=0),

        # Header
        html.H2("Dashboard de Predicción de Retrasos de Vuelos",
                className="text-white fw-bold mb-1"),
        html.P("Este sistema utiliza computación paralela y CatBoost para analizar retrasos en tiempo real.",
               className="text-muted mb-4"),
        dbc.Row([
            # Sidebar
            dbc.Col(sidebar, width=2),
            # Main content
            dbc.Col(
                dbc.Tabs(
                    [tab_datos, tab_rendimiento, tab_prediccion],
                    id="tabs",
                    active_tab="tab-datos",
                ),
                width=10,
            ),
        ]),
    ],
)

@app.callback(
    Output("resultado-prediccion", "children"),
    Input("btn-predecir", "n_clicks"),
    State("origen",      "value"),
    State("destino",     "value"),
    State("dia",         "value"),
    State("hora-vuelo",  "value"),
    State("temp",        "value"),
    State("humedad",     "value"),
    State("viento",      "value"),
    State("visibilidad", "value"),
    prevent_initial_call=True,
)
def predecir(n_clicks, origen, destino, dia, hora_vuelo, temp, humedad, viento, visibilidad):
    # Build input row
    datos = {
        "dep_iata":           origen,
        "arr_iata":           destino,
        "ruta":               f"{origen}-{destino}",
        "weather_temp":       temp,
        "weather_temp_max":   temp + 2,
        "weather_temp_min":   temp - 2,
        "delta_temp":         4.0,
        "weather_pressure":   1013,
        "weather_humidity":   humedad,
        "weather_visibility": visibilidad,
        "weather_wind_speed": viento,
        "weather_rain_1h":    0.0,
        "weather_clouds":     20,
        "hora":               hora_vuelo,
        "dia_semana":         dia,
        "hora_pico":          1 if hora_vuelo in [7, 8, 17, 18] else 0,
        "vuelo_nocturno":     1 if hora_vuelo > 20 or hora_vuelo < 6 else 0,
        "clima_severo":       1 if viento > 40 or visibilidad < 1000 else 0,
        "humedad_alta":       1 if humedad > 80 else 0,
    }
    input_df = pd.DataFrame([datos])
    prob_retraso = modelo.predict_proba(input_df)[0][1]

    # Gauge chart
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob_retraso * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar":  {"color": "#e74c3c" if prob_retraso > 0.30 else "#2ecc71"},
            "steps": [
                {"range": [0,  30], "color": "rgba(46,204,113,0.15)"},
                {"range": [30, 100], "color": "rgba(231,76,60,0.15)"},
            ],
            "threshold": {
                "line":  {"color": "orange", "width": 3},
                "thickness": 0.75,
                "value": 30,
            },
        },
        title={"text": "Probabilidad de Retraso", "font": {"color": "white"}},
    ))
    gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=260,
        margin=dict(t=40, b=10, l=30, r=30),
    )

    # Alert banner
    if prob_retraso > 0.30:
        banner = dbc.Alert(
            [
                html.Strong(f"ALTA PROBABILIDAD DE RETRASO: {prob_retraso:.1%}"),
                html.Br(),
                "Se recomienda monitorear el estado del vuelo. Los factores climáticos o la ruta indican posibles demoras.",
            ],
            color="danger",
        )
    else:
        banner = dbc.Alert(
            [
                html.Strong(f"✅  VUELO A TIEMPO — Probabilidad de retraso: {prob_retraso:.1%}"),
                html.Br(),
                "Las condiciones actuales son favorables para una salida puntual.",
            ],
            color="success",
        )

    return dbc.Row([
        dbc.Col(dcc.Graph(figure=gauge, config={"displayModeBar": False}), width=5),
        dbc.Col(banner, width=7, className="d-flex align-items-center"),
    ])

if __name__ == "__main__":
    app.run(debug=True)