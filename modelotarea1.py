# Predicción de retrasos de vuelos con CatBoost
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import shap
from numba import njit, prange


plt.style.use("ggplot")
sns.set_theme(style="whitegrid")
df = pd.read_csv("data/vuelos_clima_1778363496.csv")
df["dep_time"] = pd.to_datetime(df["dep_time"])
df["dep_estimated"] = pd.to_datetime(df["dep_estimated"])
df["arr_time"] = pd.to_datetime(df["arr_time"])

# Se trabaja la variable objetivo delayed en minutos
# Se trabaja dichos campos como campos de tipo dato
df["delayed"] = pd.to_numeric(
    df["delayed"],
    errors="coerce"
)

df["delayed"] = df["delayed"].fillna(0)

# Se realiza una estandarización de los tipos de datos
# Ademas, de un manejo de campos nulos
@njit(parallel=True)
def calcular_retrasado(delayed):
    resultado = np.zeros(delayed.shape[0], dtype=np.int32)

    for i in prange(delayed.shape[0]):
        if delayed[i] > 10:
            resultado[i] = 1
        else:
            resultado[i] = 0

    return resultado


df["retrasado"] = calcular_retrasado(
    df["delayed"].values.astype(np.float64)
)

print(df["retrasado"].value_counts())

# Se crea la variable objetivo retrasado
df["hora"] = df["dep_time"].dt.hour

df["dia_semana"] = df["dep_time"].dt.dayofweek

df["ruta"] = (
    df["dep_iata"] + "_" + df["arr_iata"]
)

@njit(parallel=True)
def calcular_hora_pico(horas):
    resultado = np.zeros(horas.shape[0], dtype=np.int32)

    for i in prange(horas.shape[0]):

        if (
            (6 <= horas[i] <= 9) or
            (17 <= horas[i] <= 20)
        ):
            resultado[i] = 1

    return resultado


@njit(parallel=True)
def calcular_vuelo_nocturno(horas):
    resultado = np.zeros(horas.shape[0], dtype=np.int32)

    for i in prange(horas.shape[0]):

        if (
            (horas[i] >= 22) or
            (horas[i] <= 5)
        ):
            resultado[i] = 1

    return resultado


df["hora_pico"] = calcular_hora_pico(
    df["hora"].values.astype(np.int32)
)

df["vuelo_nocturno"] = calcular_vuelo_nocturno(
    df["hora"].values.astype(np.int32)
)

# Se realiza una ingenieria de características para crear nuevas variables
# Estas se encuentran relacionadas con los componentes de la fecha y hora
@njit(parallel=True)
def calcular_delta_temp(temp_max, temp_min):
    resultado = np.zeros(temp_max.shape[0], dtype=np.float64)

    for i in prange(temp_max.shape[0]):
        resultado[i] = temp_max[i] - temp_min[i]

    return resultado


df["delta_temp"] = calcular_delta_temp(
    df["weather_temp_max"].values.astype(np.float64),
    df["weather_temp_min"].values.astype(np.float64)
)

@njit(parallel=True)
def calcular_clima_severo(
    visibility,
    wind_speed,
    clouds
):
    resultado = np.zeros(
        visibility.shape[0],
        dtype=np.int32
    )

    for i in prange(visibility.shape[0]):

        if (
            (visibility[i] < 3000) or
            (wind_speed[i] > 12) or
            (clouds[i] > 90)
        ):
            resultado[i] = 1

    return resultado


@njit(parallel=True)
def calcular_humedad_alta(humidity):
    resultado = np.zeros(
        humidity.shape[0],
        dtype=np.int32
    )

    for i in prange(humidity.shape[0]):

        if humidity[i] > 85:
            resultado[i] = 1

    return resultado


df["clima_severo"] = calcular_clima_severo(
    df["weather_visibility"].values.astype(np.float64),
    df["weather_wind_speed"].values.astype(np.float64),
    df["weather_clouds"].values.astype(np.float64)
)

df["humedad_alta"] = calcular_humedad_alta(
    df["weather_humidity"].values.astype(np.float64)
)

# Se realiza una ingenieria de características para crear nuevas variables
# Estas se encuentran relacionadas con los componentes del clima
df = df.dropna(subset=[
    "dep_iata",
    "arr_iata"
])


features = [
    "dep_iata",
    "arr_iata",
    "ruta",
    "weather_temp",
    "weather_temp_max",
    "weather_temp_min",
    "delta_temp",
    "weather_pressure",
    "weather_humidity",
    "weather_visibility",
    "weather_wind_speed",
    "weather_rain_1h",
    "weather_clouds",
    "hora",
    "dia_semana",
    "hora_pico",
    "vuelo_nocturno",
    "clima_severo",
    "humedad_alta"
]
X = df[features]
y = df["retrasado"]

# Se definen las caracteristicas consideradas y la variable objetivo
categoricas = [
    "dep_iata",
    "arr_iata",
    "ruta"
]


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,
    stratify=y,
    random_state=42
)

# Se realiza una división de los datos en conjuntos de entrenamiento y prueba
modelo_base = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="F1",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=0
)


param_dist = {

    "iterations": [400, 600, 800, 1000],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "depth": [4, 6, 8, 10],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
    "border_count": [32, 64, 128],
    "bagging_temperature": [0, 1, 3, 5],
    "random_strength": [1, 3, 5]
}

# Se define el modelo base de CatBoost y el espacio de búsqueda para la optimización de hiperparámetros
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


random_search = RandomizedSearchCV(
    estimator=modelo_base,
    param_distributions=param_dist,
    n_iter=40,
    scoring="f1",
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Se realiza la optimización de hiperparámetros utilizando RandomizedSearchCV con validación cruzada estratificada
random_search.fit(

    X_train,

    y_train,

    cat_features=categoricas
)
modelo = random_search.best_estimator_
print("\n==============================")
print("MEJORES PARÁMETROS")
print("==============================")
print(random_search.best_params_)

# Se entrena el modelo con los mejores hiperparámetros encontrados
# Se muestra los mejores parámetros encontrados en la optimización
y_prob = modelo.predict_proba(X_test)[:, 1]
threshold = 0.30

@njit(parallel=True)
def aplicar_threshold(probabilidades, threshold):
    resultado = np.zeros(
        probabilidades.shape[0],
        dtype=np.int32
    )

    for i in prange(probabilidades.shape[0]):

        if probabilidades[i] > threshold:
            resultado[i] = 1

    return resultado


y_pred = aplicar_threshold(
    y_prob.astype(np.float64),
    threshold
)

# Se ajusta el umbral de clasificación para mejorar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)


print("\n==============================")
print("MÉTRICAS")
print("==============================")
print(f"Threshold : {threshold}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")
print("\n==============================")
print("REPORTE")
print("==============================")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# Se muestran las métricas de rendimiento del modelo y el reporte de clasificación por clase
cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# Se muestra la matriz de confusión para visualizar el rendimiento del modelo 
# Obtener la importancia de cada variable del modelo entrenado
importancias = modelo.get_feature_importance()

feat_imp = pd.DataFrame({
    "Feature": features,
    "Importance": importancias
})

# Ordenar las variables por importancia descendente
feat_imp = feat_imp.sort_values(
    by="Importance",
    ascending=False
)


print("\n==============================")
print("IMPORTANCIA VARIABLES")
print("==============================")

print(feat_imp)

# Graficar la importancia de las variables
plt.figure(figsize=(12, 8))
sns.barplot(
    data=feat_imp,
    x="Importance",
    y="Feature"
)

plt.title("Importancia de Variables")
plt.show()


# Calcular y mostrar los valores SHAP para interpretar el modelo
print("\n==============================")
print("GENERANDO SHAP VALUES")
print("==============================")

explainer = shap.TreeExplainer(modelo)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar"
)


# Crear carpeta models en caso de no existir
os.makedirs(
    "models",
    exist_ok=True
)

# Guardar el modelo entrenado dentro de la carpeta models
modelo.save_model(
    "models/modelotarea1.cbm"
)
print("\nModelo guardado correctamente en models/modelotarea1.cbm")