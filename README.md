# Proyecto Parcial 2 — Cómputo Paralelo y Distribuido

Proyecto desarrollado en Python para la implementación del manejo de cómputo en paralelo.  
Mediante el uso de peticiones a APIs se realiza un procesamiento de datos para alimentar un modelo de predicción, con la finalidad de calcular el tiempo de retraso que un avión puede llegar a tener. El modelo utilizado considera tanto datos de la aerolínea como factores meteorológicos. Además, se permite al usuario interactuar con el modelo mediante un dashboard interactivo, así como monitorear variables estadísticas.

---

# APIs utilizadas

##  AirLabs API

La API de AirLabs se utiliza para obtener información de vuelos en tiempo real, incluyendo horarios, estados de vuelo, aeropuertos de salida y llegada, así como indicadores de retraso.

 https://airlabs.co

---

## OpenWeatherMap API

La API de OpenWeatherMap permite consultar información meteorológica asociada a cada aeropuerto, integrando variables climáticas relevantes para el análisis predictivo.

 https://openweathermap.org/api

# Variables recolectadas

## Información de vuelos

El dataset generado integra variables aeronáuticas y meteorológicas que posteriormente pueden utilizarse en modelos de Machine Learning y análisis estadístico.

- `flight_iata`
- `dep_iata`
- `dep_time`
- `dep_estimated`
- `arr_iata`
- `arr_time`
- `status`
- `delayed`

---

## Información climática

- `weather_temp`
- `weather_temp_max`
- `weather_temp_min`
- `weather_pressure`
- `weather_humidity`
- `weather_sea_level`
- `weather_visibility`
- `weather_wind_speed`
- `weather_rain_1h`
- `weather_clouds`

---

# Procesamiento concurrente

El sistema implementa control de concurrencia mediante un semáforo:

```python
semaforo = asyncio.Semaphore(10)
```

Esto permite limitar el número de solicitudes simultáneas, evitando saturar las APIs y mejorando la estabilidad del sistema durante la ejecución.

---

Para ejecutar el proyecto es necesario crear un archivo `.env` en la raíz del proyecto con las credenciales correspondientes de ambas APIs.

```env
AIRLABS_API_KEY=TU_API_KEY
OPENWEATHER_API_KEY=TU_API_KEY
```
---

# Archivo de salida

Los datos obtenidos se almacenan automáticamente dentro de la carpeta `data` en formato CSV.

```text
data/vuelos_clima_TIMESTAMP.csv
```

Cada archivo generado contiene la información recopilada durante la ejecución del sistema.

---
# Modelo de Predicción de Retrasos de Vuelos con CatBoost

El proyecto incorpora un modelo de Machine Learning orientado a la predicción de retrasos de vuelos utilizando `CatBoostClassifier`. El objetivo principal es identificar si un vuelo presentará retrasos significativos considerando tanto variables operacionales como factores meteorológicos.

El modelo fue construido a partir del dataset generado mediante el sistema de recolección asíncrona de datos, integrando información de vuelos y condiciones climáticas en tiempo real.

---

# Procesamiento y preparación de datos

El sistema realiza una etapa de preprocesamiento donde las variables temporales son convertidas a formatos de fecha y hora para facilitar la ingeniería de características.

Posteriormente, la variable objetivo `delayed` es transformada a formato numérico y utilizada para generar una nueva variable binaria denominada `retrasado`, la cual identifica vuelos con retrasos mayores a 10 minutos.

Para optimizar el rendimiento computacional, varias operaciones fueron implementadas utilizando `Numba` mediante decoradores `@njit(parallel=True)` y procesamiento paralelo con `prange`.

---

# Ingeniería de características

El modelo incorpora múltiples variables derivadas para mejorar la capacidad predictiva. Entre las principales transformaciones realizadas se encuentran:

- Extracción de hora del vuelo
- Día de la semana
- Construcción de rutas aeropuerto-origen/destino
- Identificación de horas pico
- Identificación de vuelos nocturnos
- Cálculo de variación de temperatura
- Detección de clima severo
- Identificación de humedad alta

Estas variables permiten representar de mejor manera patrones operacionales y condiciones ambientales asociadas a retrasos.

---

# Variables utilizadas

## Variables categóricas

- `dep_iata`
- `arr_iata`
- `ruta`

## Variables numéricas y climáticas

- `weather_temp`
- `weather_temp_max`
- `weather_temp_min`
- `delta_temp`
- `weather_pressure`
- `weather_humidity`
- `weather_visibility`
- `weather_wind_speed`
- `weather_rain_1h`
- `weather_clouds`
- `hora`
- `dia_semana`
- `hora_pico`
- `vuelo_nocturno`
- `clima_severo`
- `humedad_alta`

---

# Modelo utilizado

El sistema utiliza `CatBoostClassifier`, un algoritmo basado en gradient boosting especializado en el manejo eficiente de variables categóricas.

El modelo fue configurado utilizando:

```python
CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="F1",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=0
)
```

---

# Optimización de hiperparámetros

Para mejorar el desempeño del modelo se implementó una búsqueda aleatoria de hiperparámetros mediante `RandomizedSearchCV` utilizando validación cruzada estratificada (`StratifiedKFold`).

La optimización considera parámetros como:

- número de iteraciones,
- profundidad de árboles,
- learning rate,
- regularización,
- bagging,
- random strength.

Esto permite encontrar configuraciones más robustas para el problema de clasificación.

---

# Evaluación del modelo

El desempeño del modelo es evaluado utilizando distintas métricas de clasificación:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

Además, se genera un reporte completo de clasificación y una matriz de confusión para analizar el comportamiento del modelo en cada clase.

---

# Interpretabilidad del modelo

Para analizar la importancia de las variables se utilizan dos enfoques:

## Importancia de variables

El modelo calcula automáticamente la relevancia de cada característica utilizada durante el entrenamiento.

## SHAP Values

Se implementa `SHAP` para interpretar el impacto individual de cada variable sobre las predicciones del modelo, permitiendo comprender mejor el comportamiento interno del algoritmo.

---

# Procesamiento paralelo

El proyecto utiliza paralelización mediante `Numba` para acelerar operaciones de ingeniería de características y procesamiento de datos.

Funciones implementadas con:

```python
@njit(parallel=True)
```

permiten optimizar:

- creación de variables binarias,
- cálculo de clima severo,
- detección de horas pico,
- cálculo de métricas derivadas.

---

# Visualización de resultados

El sistema genera distintas visualizaciones para facilitar el análisis del modelo:

- Matriz de confusión
- Gráfica de importancia de variables
- SHAP Summary Plot

Estas visualizaciones permiten evaluar tanto el rendimiento como la interpretabilidad del sistema predictivo.

---

# Almacenamiento del modelo

El modelo entrenado se almacena automáticamente dentro de la carpeta `models` utilizando el formato `.cbm` de CatBoost.

```python
modelo.save_model(
    "models/modelotarea1.cbm"
)
```

---
