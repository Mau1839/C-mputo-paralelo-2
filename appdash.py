import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from catboost import CatBoostClassifier
import os
import subprocess
import sys

# ── Auto-ejecución de data_engine_mpi.py si no existe la caché ──
def ejecutar_mpi():
    cache_path = "data/mpi_cache_retrasos.csv"
    if not os.path.exists(cache_path):
        print("[AeroBI] Caché MPI no encontrada. Ejecutando data_engine_mpi.py con 4 procesos...")
        try:
            resultado = subprocess.run(
                ["mpirun", "-n", "4", sys.executable, "data_engine_mpi.py"],
                check=True,
                capture_output=False,
            )
            print("[AeroBI] data_engine_mpi.py completado exitosamente.")
        except subprocess.CalledProcessError as e:
            print(f"[AeroBI] ERROR al ejecutar MPI: {e}")
            print("[AeroBI] Continuando sin caché MPI...")
        except FileNotFoundError:
            print("[AeroBI] ERROR: 'mpirun' no encontrado. Verifica tu instalación de MPI.")
    else:
        print(f"[AeroBI] Caché MPI encontrada. Saltando ejecución de data_engine_mpi.py.")

ejecutar_mpi()

# Encapsulamiento de estilos

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body {
    background: #0A0E1A !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #F1F5F9 !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1F2D45; border-radius: 3px; }
/* Firefox */
html { scrollbar-width: thin; scrollbar-color: #1F2D45 #0A0E1A; }

/* ── Sidebar ── */
.sidebar-card {
    background: #111827;
    border: 1px solid #1F2D45;
    border-radius: 16px;
    padding: 24px 20px;
    height: 100%;
}

/* ── Metric cards ── */
.metric-card {
    background: #111827;
    border: 1px solid #1F2D45;
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
}
.metric-card:hover { border-color: #3B82F6; }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3B82F6, #06B6D4);
    border-radius: 12px 12px 0 0;
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #F1F5F9;
    font-family: 'DM Mono', monospace;
    letter-spacing: -.02em;
}

/* ── Tabs ── */
.nav-tabs { border-bottom: 1px solid #1F2D45 !important; }
.nav-tabs .nav-link {
    color: #64748B !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .04em;
    padding: 12px 20px !important;
    transition: color .2s, border-color .2s;
}
.nav-tabs .nav-link:hover { color: #94A3B8 !important; }
.nav-tabs .nav-link.active {
    color: #3B82F6 !important;
    border-bottom: 2px solid #3B82F6 !important;
}
.tab-content { padding-top: 24px; }

/* ── Main content card ── */
.main-card {
    background: #111827;
    border: 1px solid #1F2D45;
    border-radius: 16px;
    padding: 28px;
}

/* ── Divider ── */
.divider { border-color: #1F2D45 !important; margin: 20px 0; }

/* ── Buttons ── */
.btn-predict {
    background: linear-gradient(135deg, #3B82F6, #06B6D4) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: .04em !important;
    padding: 14px !important;
    color: #fff !important;
    transition: opacity .2s, transform .1s !important;
}
.btn-predict:hover { opacity: .9 !important; transform: translateY(-1px) !important; }
.btn-predict:active { transform: translateY(0) !important; }

/* ── Form labels ── */
.form-label-pro {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 8px;
    display: block;
}

/* ── Inputs ── */
.pro-input {
    background: #0A0E1A !important;
    border: 1px solid #1F2D45 !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    transition: border-color .2s;
}
.pro-input:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.15) !important;
    outline: none !important;
}
.form-control { background: #0A0E1A !important; border: 1px solid #1F2D45 !important;
    color: #F1F5F9 !important; border-radius: 8px !important; }
.form-control:focus { border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.15) !important; }

/* Dash v2 react-select overrides */
.dash-dropdown .Select__control {
    background: #0A0E1A !important; border: 1px solid #1F2D45 !important; border-radius: 8px !important; }
.dash-dropdown .Select__single-value,
.dash-dropdown .Select__input-container { color: #F1F5F9 !important; }
.dash-dropdown .Select__placeholder { color: #64748B !important; }
.dash-dropdown .Select__menu { background: #1C2333 !important;
    border: 1px solid #1F2D45 !important; border-radius: 8px !important; }
.dash-dropdown .Select__option { color: #94A3B8 !important; }
.dash-dropdown .Select__option--is-focused { background: #1F2D45 !important; color: #F1F5F9 !important; }
.dash-dropdown .Select__option--is-selected { background: rgba(59,130,246,.2) !important;
    color: #3B82F6 !important; }

/* ── Slider ── */
.rc-slider-rail { background: #1F2D45 !important; }
.rc-slider-track { background: linear-gradient(90deg,#3B82F6,#06B6D4) !important; }
.rc-slider-handle { border-color: #3B82F6 !important; background: #3B82F6 !important; }
.rc-slider-mark-text { color: #64748B !important; font-size: 11px !important; }

/* ── Alert overrides ── */
.alert-info { background: rgba(6,182,212,.1) !important; border: 1px solid rgba(6,182,212,.25) !important;
    color: #94A3B8 !important; border-radius: 10px !important; }
.alert-warning { background: rgba(245,158,11,.1) !important; border: 1px solid rgba(245,158,11,.3) !important;
    color: #F59E0B !important; border-radius: 10px !important; font-family: 'DM Mono', monospace;
    font-size: 13px; text-align: center; letter-spacing: .04em; }
.alert-danger { background: rgba(239,68,68,.1) !important; border: 1px solid rgba(239,68,68,.25) !important;
    color: #F1F5F9 !important; border-radius: 10px !important; }
.alert-success { background: rgba(16,185,129,.1) !important; border: 1px solid rgba(16,185,129,.25) !important;
    color: #F1F5F9 !important; border-radius: 10px !important; }

/* ── Section header ── */
.section-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #3B82F6;
    margin-bottom: 16px;
}

/* ── Status chip ── */
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16,185,129,.12);
    border: 1px solid rgba(16,185,129,.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
    color: #10B981;
    letter-spacing: .06em;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #10B981;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: .3; }
}

/* ── Header ── */
.app-header {
    border-bottom: 1px solid #1F2D45;
    padding-bottom: 20px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.app-title {
    font-size: 20px;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -.02em;
    margin: 0;
}
.app-subtitle {
    font-size: 13px;
    color: #64748B;
    margin: 4px 0 0 0;
}

/* ── Sidebar stat row ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #1F2D45;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { font-size: 12px; color: #64748B; font-weight: 500; }
.stat-val { font-size: 13px; color: #F1F5F9; font-weight: 600; font-family: 'DM Mono', monospace; }

/* ── Graph containers ── */
.graph-card {
    background: #0A0E1A;
    border: 1px solid #1F2D45;
    border-radius: 12px;
    padding: 4px;
}
"""

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94A3B8", size=12),
    xaxis=dict(gridcolor="#1F2D45", zerolinecolor="#1F2D45", tickfont=dict(color="#64748B", size=11)),
    yaxis=dict(gridcolor="#1F2D45", zerolinecolor="#1F2D45", tickfont=dict(color="#64748B", size=11)),
    margin=dict(t=56, b=24, l=24, r=24),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8")),
    coloraxis=dict(colorbar=dict(tickfont=dict(color="#64748B"))),
)

def pro_title(text):
    return dict(text=text, font=dict(color="#F1F5F9", size=14, family="DM Sans"), x=0.02, xanchor="left")

BLUE_CYAN = ["#1E3A5F", "#1E4976", "#1A5FAD", "#3B82F6", "#60A5FA", "#06B6D4", "#67E8F9"]

# Arranque de la App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Flight Intelligence · AeroBI")
server = app.server

# ── Inyectar CSS personalizado en el <head> (sin esto no hay estilos) ──
_orig = app.index_string
app.index_string = _orig.replace("</head>", f"<style>{CUSTOM_CSS}</style></head>")

# --- INTEGRACIÓN MPI ---
def cargar_datos_mpi():
    if os.path.exists("data/mpi_cache_retrasos.csv"):
        return pd.read_csv("data/mpi_cache_retrasos.csv")
    else:
        print("Caché MPI no encontrada. Por favor ejecuta mpirun primero.")
        return pd.DataFrame(columns=["dep_iata", "retraso_promedio", "retraso_maximo"])

df_completo = pd.read_csv("data/vuelos_clima_1778363496.csv")
df_mpi_cache = cargar_datos_mpi()

def cargar_modelo():
    m = CatBoostClassifier()
    m.load_model("models/modelotarea1.cbm")
    return m

modelo = cargar_modelo()

FEATURES = [
    "dep_iata", "arr_iata", "ruta", "weather_temp", "weather_temp_max",
    "weather_temp_min", "delta_temp", "weather_pressure", "weather_humidity",
    "weather_visibility", "weather_wind_speed", "weather_rain_1h",
    "weather_clouds", "hora", "dia_semana", "hora_pico",
    "vuelo_nocturno", "clima_severo", "humedad_alta",
]

def metric_card(label, value, icon=""):
    return html.Div([
        html.Div(f"{icon}  {label}".strip(), className="metric-label"),
        html.Div(value, className="metric-value"),
    ], className="metric-card")

# --- GRÁFICA ALIMENTADA POR MPI ---
retrasos_apt = (
    df_mpi_cache
    .sort_values("retraso_promedio", ascending=False)
    .head(15)
)

fig_airports = go.Figure(go.Bar(
    x=retrasos_apt["dep_iata"],
    y=retrasos_apt["retraso_promedio"],
    marker=dict(
        color=retrasos_apt["retraso_promedio"],
        colorscale=BLUE_CYAN, line=dict(width=0),
    ),
    hovertemplate="<b>%{x}</b><br>Retraso Prom: %{y:.1f} min<extra></extra>",
))
fig_airports.update_layout(**BASE_LAYOUT, title=pro_title("Top 15 Aeropuertos (Procesado vía MPI)"), bargap=0.35)

# Cargar Características del Modelo
importancias = modelo.get_feature_importance()
feat_imp = pd.DataFrame({"Feature": FEATURES, "Importance": importancias}).sort_values("Importance", ascending=True).tail(10)

fig_imp = go.Figure(go.Bar(
    x=feat_imp["Importance"], y=feat_imp["Feature"], orientation="h",
    marker=dict(color=feat_imp["Importance"], colorscale=BLUE_CYAN, line=dict(width=0)),
    hovertemplate="<b>%{y}</b><br>Importancia: %{x:.2f}<extra></extra>",
))
fig_imp.update_layout(**BASE_LAYOUT, title=pro_title("Importancia de Variables"), xaxis_title="", yaxis_title="", bargap=0.35)

# ── FIX 1: usar df_completo['delayed'] en lugar de df['delayed'] ──
retraso_global_promedio = df_completo['delayed'].mean() if 'delayed' in df_completo.columns else 0
retraso_global_max = df_mpi_cache['retraso_maximo'].max() if not df_mpi_cache.empty else 0

tab_datos = dbc.Tab(
    label="Exploración", tab_id="tab-datos",
    children=[
        dbc.Row([
            dbc.Col(metric_card("Retraso Promedio", f"{retraso_global_promedio:.1f} min"), md=3),
            dbc.Col(metric_card("Retraso Máximo",   f"{int(retraso_global_max)} min"),  md=3),
            dbc.Col(metric_card("Vuelos Analizados", f"{len(df_completo):,}"), md=3),
            # ── FIX 2: df_completo['dep_iata'].nunique() en vez de dep_iata.nunique() ──
            dbc.Col(metric_card("Aeropuertos", str(df_completo['dep_iata'].nunique())), md=3),
        ], className="g-3 mb-4"),
        html.Hr(className="divider"),
        html.Div(dcc.Graph(figure=fig_airports, config={"displayModeBar": False}, style={"height": "360px"}), className="graph-card"),
    ],
)

tab_rendimiento = dbc.Tab(
    label="Rendimiento",
    tab_id="tab-rendimiento",
    children=[
        dbc.Row([
            dbc.Col(metric_card("Accuracy",  "99.85%"), md=3),
            dbc.Col(metric_card("Precision", "99.21%"), md=3),
            dbc.Col(metric_card("Recall",    "97.35%"), md=3),
            dbc.Col(metric_card("F1-Score",  "0.9827"),  md=3),
        ], className="g-3 mb-4"),
        html.Hr(className="divider"),
        dbc.Row([
            dbc.Col(
                html.Div(
                    dcc.Graph(figure=fig_imp, config={"displayModeBar": False},
                              style={"height": "340px"}),
                    className="graph-card",
                ),
                md=8,
            ),
            dbc.Col([
                html.Div("Interpretación", className="section-title"),
                html.Div([
                    html.Div([
                        html.Span("Ruta & Aeropuerto", style={"color": "#3B82F6", "fontWeight": "600", "fontSize": "13px"}),
                        html.P("Variables categóricas con mayor poder predictivo. Las rutas históricamente congestionadas concentran la mayor parte del riesgo.", style={"fontSize": "12px", "color": "#64748B", "marginTop": "4px"}),
                    ], style={"marginBottom": "16px", "paddingBottom": "16px", "borderBottom": "1px solid #1F2D45"}),
                    html.Div([
                        html.Span("Visibilidad & Viento", style={"color": "#06B6D4", "fontWeight": "600", "fontSize": "13px"}),
                        html.P("Indicadores climáticos clave vinculados a restricciones operativas de seguridad.", style={"fontSize": "12px", "color": "#64748B", "marginTop": "4px"}),
                    ], style={"marginBottom": "16px", "paddingBottom": "16px", "borderBottom": "1px solid #1F2D45"}),
                    html.Div([
                        html.Span("Horas Pico", style={"color": "#8B5CF6", "fontWeight": "600", "fontSize": "13px"}),
                        html.P("Saturación de pistas en ventanas de máxima demanda: 7–9h y 17–19h.", style={"fontSize": "12px", "color": "#64748B", "marginTop": "4px"}),
                    ]),
                ], style={"background": "#0A0E1A", "border": "1px solid #1F2D45", "borderRadius": "12px", "padding": "20px", "marginBottom": "16px"}),
                html.Div([
                    html.Div("Umbral de Decisión", style={"fontSize": "11px", "color": "#64748B", "fontWeight": "600", "letterSpacing": ".08em", "textTransform": "uppercase"}),
                    html.Div("0.30", style={"fontSize": "32px", "fontWeight": "700", "color": "#F59E0B", "fontFamily": "'DM Mono', monospace"}),
                    html.Div("Optimizado para maximizar Recall", style={"fontSize": "11px", "color": "#64748B"}),
                ], style={"background": "rgba(245,158,11,.08)", "border": "1px solid rgba(245,158,11,.25)", "borderRadius": "12px", "padding": "20px", "textAlign": "center"}),
            ], md=4),
        ], className="g-3"),
    ],
)


def form_group(label, children):
    return html.Div([
        html.Label(label, className="form-label-pro"),
        children,
    ], style={"marginBottom": "20px"})

tab_prediccion = dbc.Tab(
    label="Simulador",
    tab_id="tab-prediccion",
    children=[
        dbc.Row([
            dbc.Col([
                html.Div("Detalles del Vuelo", className="section-title"),
                form_group("Aeropuerto de Origen",
                    dcc.Dropdown(
                        id="origen",
                        # ── FIX 3: df_completo["dep_iata"].unique() ──
                        options=[{"label": v, "value": v} for v in sorted(df_completo["dep_iata"].unique())],
                        value=df_completo["dep_iata"].iloc[0],
                        optionHeight=35,style={"color": "#212529"},
                    )
                ),
                form_group("Aeropuerto de Destino",
                    dcc.Dropdown(
                        id="destino",
                        # ── FIX 4: df_completo["arr_iata"].unique() ──
                        options=[{"label": v, "value": v} for v in sorted(df_completo["arr_iata"].unique())],
                        value=df_completo["arr_iata"].iloc[0],
                        optionHeight=35,style={"color": "#212529"},
                    )
                ),
                form_group("Día de la Semana (0=Lun … 6=Dom)",
                    dbc.Input(id="dia", type="number", min=0, max=6, step=1, value=0, className="pro-input"),
                ),
                form_group("Hora de Salida (0 – 23)",
                    dbc.Input(id="hora-vuelo", type="number", min=0, max=23, step=1, value=12, className="pro-input"),
                ),
            ], md=6),

            dbc.Col([
                html.Div("Condiciones Climáticas", className="section-title"),
                form_group("Temperatura (°C)",
                    dbc.Input(id="temp", type="number", value=20, className="pro-input"),
                ),
                form_group("Humedad Relativa (0 – 100 %)",
                    dbc.Input(id="humedad", type="number", min=0, max=100, step=1, value=50, className="pro-input"),
                ),
                form_group("Velocidad del Viento (km/h)",
                    dbc.Input(id="viento", type="number", value=10, className="pro-input"),
                ),
                form_group("Visibilidad (m)",
                    dbc.Input(id="visibilidad", type="number", value=10000, className="pro-input"),
                ),
            ], md=6),
        ], className="g-4"),

        html.Hr(className="divider"),

        dbc.Button(
            "Calcular Probabilidad de Retraso →",
            id="btn-predecir",
            size="lg",
            className="w-100 btn-predict mb-4",
        ),

        html.Div(id="resultado-prediccion"),
    ],
)


# ── FIX 5: df_completo["delayed"] y df_completo["ruta"] en sidebar ──
delayed_pct = (df_completo["delayed"] > 0).mean() * 100
rutas_unicas = df_completo["ruta"].nunique() if "ruta" in df_completo.columns else "—"

sidebar = html.Div([
    html.Div([
        html.Div(style={"width": "28px", "height": "28px", "background": "linear-gradient(135deg,#3B82F6,#06B6D4)",
                        "borderRadius": "8px", "display": "inline-block", "marginBottom": "12px"}),
        html.Div("AeroBI", style={"fontSize": "16px", "fontWeight": "700", "color": "#F1F5F9", "letterSpacing": "-.01em"}),
    ]),
    html.Div("Dataset", className="section-title"),
    html.Div([
        # ── FIX 6: len(df_completo) y df_completo['dep_iata'].nunique() ──
        html.Div([html.Span("Vuelos", className="stat-key"), html.Span(f"{len(df_completo):,}", className="stat-val")], className="stat-row"),
        html.Div([html.Span("Aeropuertos", className="stat-key"), html.Span(str(df_completo['dep_iata'].nunique()), className="stat-val")], className="stat-row"),
        html.Div([html.Span("% con Retraso", className="stat-key"), html.Span(f"{delayed_pct:.1f}%", className="stat-val")], className="stat-row"),
    ], style={"marginBottom": "24px"}),
    html.Div("Modelo", className="section-title"),
    html.Div([
        html.Div([html.Span("Algoritmo", className="stat-key"), html.Span("CatBoost", className="stat-val")], className="stat-row"),
        html.Div([html.Span("Umbral", className="stat-key"), html.Span("0.30", className="stat-val")], className="stat-row"),
        html.Div([html.Span("Versión", className="stat-key"), html.Span("v2.1", className="stat-val")], className="stat-row"),
    ], style={"marginBottom": "24px"}),
    html.Div(
        [html.Div(className="status-dot"), "Sistema Activo"],
        className="status-chip",
    ),
], className="sidebar-card")


app.layout = html.Div(
    style={"minHeight": "100vh", "background": "#0A0E1A", "padding": "32px"},
    children=[
        dcc.Interval(id="refresh-interval", interval=5 * 60 * 1000, n_intervals=0),

        html.Div([
            html.Div([
                html.H1("AeroBI | Dashboard de Predicción de Retrasos", className="app-title"),
                html.P("Modelo CatBoost · Datos meteorológicos integrados", className="app-subtitle"),
            ]),
            html.Div(
                [html.Div(className="status-dot"), "En vivo"],
                className="status-chip",
            ),
        ], className="app-header"),

        dbc.Row([
            dbc.Col(sidebar, xs=12, md=2, style={"marginBottom": "16px"}),
            dbc.Col(
                html.Div([
                    dbc.Tabs(
                        [tab_datos, tab_rendimiento, tab_prediccion],
                        id="tabs",
                        active_tab="tab-datos",
                    ),
                ], className="main-card"),
                xs=12, md=10,
            ),
        ], className="g-4"),
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
def predecir(n, origen, destino, dia, hora, temp, humedad, viento, vis):
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
        "weather_visibility": vis,
        "weather_wind_speed": viento,
        "weather_rain_1h":    0.0,
        "weather_clouds":     20,
        "hora":               hora,
        "dia_semana":         dia,
        "hora_pico":          1 if hora in [7, 8, 17, 18] else 0,
        "vuelo_nocturno":     1 if hora > 20 or hora < 6 else 0,
        "clima_severo":       1 if viento > 40 or vis < 1000 else 0,
        "humedad_alta":       1 if humedad > 80 else 0,
    }
    prob = modelo.predict_proba(pd.DataFrame([datos]))[0][1]
    is_delayed = prob > 0.30

    gauge_color = "#EF4444" if is_delayed else "#10B981"
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 44, "color": "#F1F5F9", "family": "DM Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#1F2D45", "tickfont": {"color": "#64748B", "size": 11}},
            "bar":  {"color": gauge_color, "thickness": 0.22},
            "bgcolor": "#0A0E1A",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(16,185,129,.08)"},
                {"range": [30, 100], "color": "rgba(239,68,68,.08)"},
            ],
            "threshold": {
                "line":  {"color": "#F59E0B", "width": 2},
                "thickness": 0.8,
                "value": 30,
            },
        },
        domain={"x": [0.05, 0.95], "y": [0, 1]},
    ))
    gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(t=20, b=10, l=20, r=20),
    )

    if is_delayed:
        result_content = html.Div([
            html.Div("⚠ ALERTA DE RETRASO", style={
                "fontSize": "11px", "fontWeight": "700", "letterSpacing": ".12em",
                "color": "#EF4444", "marginBottom": "8px",
            }),
            html.Div(f"{prob:.1%}", style={
                "fontSize": "48px", "fontWeight": "700",
                "color": "#EF4444", "fontFamily": "'DM Mono', monospace",
                "lineHeight": "1", "marginBottom": "8px",
            }),
            html.Div("Probabilidad de retraso estimada", style={"fontSize": "13px", "color": "#64748B", "marginBottom": "16px"}),
            html.Div(f"Ruta {origen} → {destino}", style={
                "fontSize": "12px", "color": "#94A3B8", "background": "#1C2333",
                "padding": "8px 14px", "borderRadius": "8px",
                "fontFamily": "'DM Mono', monospace", "marginBottom": "12px",
            }),
            html.P("Se recomienda monitorear activamente el estado del vuelo. Las condiciones actuales de ruta o clima elevan el riesgo operativo por encima del umbral de decisión.",
                   style={"fontSize": "13px", "color": "#94A3B8", "lineHeight": "1.6"}),
        ], style={
            "background": "rgba(239,68,68,.07)", "border": "1px solid rgba(239,68,68,.25)",
            "borderRadius": "14px", "padding": "28px", "height": "100%",
        })
    else:
        result_content = html.Div([
            html.Div("✓ VUELO EN TIEMPO", style={
                "fontSize": "11px", "fontWeight": "700", "letterSpacing": ".12em",
                "color": "#10B981", "marginBottom": "8px",
            }),
            html.Div(f"{prob:.1%}", style={
                "fontSize": "48px", "fontWeight": "700",
                "color": "#10B981", "fontFamily": "'DM Mono', monospace",
                "lineHeight": "1", "marginBottom": "8px",
            }),
            html.Div("Probabilidad de retraso estimada", style={"fontSize": "13px", "color": "#64748B", "marginBottom": "16px"}),
            html.Div(f"Ruta {origen} → {destino}", style={
                "fontSize": "12px", "color": "#94A3B8", "background": "#1C2333",
                "padding": "8px 14px", "borderRadius": "8px",
                "fontFamily": "'DM Mono', monospace", "marginBottom": "12px",
            }),
            html.P("Las condiciones actuales son favorables para una salida puntual. El modelo no detecta factores de riesgo significativos en esta operación.",
                   style={"fontSize": "13px", "color": "#94A3B8", "lineHeight": "1.6"}),
        ], style={
            "background": "rgba(16,185,129,.07)", "border": "1px solid rgba(16,185,129,.25)",
            "borderRadius": "14px", "padding": "28px", "height": "100%",
        })

    return dbc.Row([
        dbc.Col(
            html.Div(dcc.Graph(figure=gauge, config={"displayModeBar": False}),
                     style={"background": "#0A0E1A", "border": "1px solid #1F2D45", "borderRadius": "14px"}),
            md=5,
        ),
        dbc.Col(result_content, md=7),
    ], className="g-3")


if __name__ == "__main__":
    app.run(debug=True)
