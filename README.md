# Proyecto Parcial 2 — Cómputo Paralelo y Distribuido

Proyecto desarrollado en Python para la implementación de técnicas de cómputo paralelo y distribuido aplicadas al análisis predictivo de retrasos de vuelos. Mediante el uso de peticiones concurrentes a APIs externas se realiza una recolección masiva de datos aeronáuticos y meteorológicos para alimentar un modelo de Machine Learning orientado a la predicción de retrasos.

El sistema integra procesamiento paralelo, optimización computacional, análisis estadístico y visualización interactiva mediante un dashboard desarrollado con Dash y Plotly.

---

# APIs utilizadas

## AirLabs API

La API de AirLabs se utiliza para obtener información de vuelos en tiempo real, incluyendo horarios, estados de vuelo, aeropuertos de salida y llegada, así como indicadores de retraso.

https://airlabs.co

---

## OpenWeatherMap API

La API de OpenWeatherMap permite consultar información meteorológica asociada a cada aeropuerto, integrando variables climáticas relevantes para el análisis predictivo.

https://openweathermap.org/api

---

# Variables recolectadas

## Información de vuelos

El dataset generado integra variables aeronáuticas y meteorológicas utilizadas posteriormente en modelos de Machine Learning y análisis estadístico.

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

El sistema implementa control de concurrencia mediante un semáforo de `asyncio` para limitar el número de solicitudes simultáneas realizadas hacia las APIs externas.

```python
semaforo = asyncio.Semaphore(10)
```

Esto permite evitar saturación de servicios, mejorar estabilidad y controlar el consumo de recursos durante la ejecución.

---

# Variables de entorno

Para ejecutar el proyecto es necesario crear un archivo `.env` en la raíz del proyecto con las credenciales correspondientes.

```env
AIRLABS_API_KEY=TU_API_KEY
OPENWEATHER_API_KEY=TU_API_KEY
```

---

# Archivo de salida

Los datos recolectados se almacenan automáticamente dentro de la carpeta `data` en formato CSV.

```text
data/vuelos_clima_TIMESTAMP.csv
```

Cada archivo contiene información consolidada de vuelos y condiciones climáticas asociadas.

---

# Procesamiento paralelo con MPI

El proyecto incorpora procesamiento distribuido mediante `MPI` utilizando la librería `mpi4py`. El objetivo es dividir el dataset entre múltiples procesos para acelerar el análisis estadístico y generación de métricas utilizadas por el dashboard.

El archivo `data_engine_mpi.py` implementa un flujo distribuido compuesto por:

- Distribución de datos mediante `scatter`
- Procesamiento paralelo por nodo
- Recolección de resultados con `gather`
- Reducción y consolidación final en el nodo maestro

---

## Flujo de procesamiento MPI

Cada proceso recibe un fragmento del dataset y calcula métricas locales relacionadas con retrasos de vuelos:

- retraso promedio,
- retraso máximo,
- vuelos retrasados por aeropuerto.

Posteriormente, el nodo maestro unifica los resultados parciales y genera un archivo cacheado optimizado para visualización.

---

## Archivo generado por MPI

```text
data/mpi_cache_retrasos.csv
```

Este archivo almacena estadísticas resumidas utilizadas posteriormente por el dashboard interactivo.

---

## Ejecución MPI

El procesamiento distribuido puede ejecutarse utilizando múltiples procesos:

```bash
mpiexec -n 4 python data_engine_mpi.py
```

Donde:

- `-n 4` indica el número de procesos paralelos.
- Cada nodo procesa una parte distinta del dataset.

---

# Modelo de Predicción de Retrasos de Vuelos con CatBoost

El proyecto incorpora un modelo de Machine Learning orientado a la predicción de retrasos de vuelos utilizando `CatBoostClassifier`.

El objetivo principal es identificar vuelos con probabilidad alta de retraso considerando variables operacionales y meteorológicas.

---

# Procesamiento y preparación de datos

El sistema realiza una etapa de preprocesamiento donde las variables temporales son convertidas a formatos adecuados para análisis.

La variable objetivo `delayed` es transformada a formato numérico y utilizada para construir una nueva variable binaria denominada `retrasado`, la cual identifica vuelos con retrasos mayores a 10 minutos.

Para optimizar el rendimiento computacional se implementaron funciones paralelizadas utilizando `Numba`.

```python
@njit(parallel=True)
```

---

# Ingeniería de características

El modelo incorpora múltiples variables derivadas:

- Hora del vuelo
- Día de la semana
- Ruta origen-destino
- Hora pico
- Vuelo nocturno
- Delta de temperatura
- Clima severo
- Humedad alta

Estas variables permiten representar mejor patrones operacionales y ambientales asociados a retrasos.

---

# Variables utilizadas

## Variables categóricas

- `dep_iata`
- `arr_iata`
- `ruta`

---

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

El sistema utiliza `CatBoostClassifier`, algoritmo basado en Gradient Boosting especializado en variables categóricas.

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

Se implementa optimización mediante `RandomizedSearchCV` y validación cruzada estratificada utilizando `StratifiedKFold`.

La búsqueda considera parámetros relacionados con:

- número de iteraciones,
- profundidad de árboles,
- learning rate,
- regularización,
- bagging,
- random strength.

---

# Evaluación del modelo

El modelo es evaluado utilizando distintas métricas de clasificación:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

También se generan:

- matriz de confusión,
- reporte de clasificación,
- análisis de desempeño por clase.

---

# Interpretabilidad del modelo

Para interpretar el comportamiento del modelo se utilizan:

## Importancia de variables

El modelo calcula automáticamente la relevancia de cada característica.

## SHAP Values

Se implementa `SHAP` para analizar el impacto individual de las variables sobre las predicciones realizadas.

---

# Procesamiento paralelo con Numba

El proyecto utiliza `Numba` para acelerar operaciones de ingeniería de características y procesamiento numérico.

Funciones implementadas con:

```python
@njit(parallel=True)
```

permiten optimizar:

- creación de variables binarias,
- detección de clima severo,
- identificación de horas pico,
- cálculo de métricas derivadas.

---

# Visualización de resultados

El sistema genera distintas visualizaciones para facilitar el análisis:

- Matriz de confusión
- Gráfica de importancia de variables
- SHAP Summary Plot

Estas visualizaciones permiten evaluar rendimiento e interpretabilidad del modelo.

---

# Almacenamiento del modelo

El modelo entrenado se almacena automáticamente dentro de la carpeta `models`.

```python
modelo.save_model(
    "models/modelotarea1.cbm"
)
```

---

# Dashboard Interactivo

El proyecto incluye un dashboard interactivo desarrollado con Dash, Plotly y Bootstrap para visualizar métricas, analizar información histórica y realizar simulaciones de retrasos en tiempo real.

El sistema carga automáticamente el modelo entrenado y permite interacción dinámica mediante una interfaz moderna enfocada en análisis operacional.

---

# Funcionalidades principales

- Visualización de métricas generales del dataset.
- Retraso promedio y máximo.
- Número de vuelos analizados.
- Número de aeropuertos procesados.
- Visualización de aeropuertos con mayor retraso promedio.
- Importancia de variables del modelo.
- Simulador de predicción de retrasos.
- Indicadores visuales dinámicos.
- Gauge chart de probabilidad de retraso.
- Alertas de riesgo operacional.
- Dashboard responsivo con CSS personalizado.

---

# Variables utilizadas por el simulador

## Información del vuelo

- Aeropuerto de origen
- Aeropuerto de destino
- Ruta
- Hora del vuelo
- Día de la semana

---

## Variables meteorológicas

- Temperatura
- Humedad
- Velocidad del viento
- Visibilidad
- Lluvia
- Nubosidad
- Presión atmosférica

---

## Variables derivadas

- Hora pico
- Vuelo nocturno
- Clima severo
- Humedad alta
- Delta de temperatura

---

# Visualización y análisis

El dashboard integra gráficos avanzados utilizando Plotly para representar:

- Importancia de características
- Distribución de retrasos por aeropuerto
- Indicadores de desempeño
- Probabilidad estimada de retraso

---

# Estructura del proyecto

```text
.
├── appdash.py
├── main.py
├── modelotarea1.py
├── data_engine_mpi.py
├── README.md
├── data/
├── models/
└── catboost_info/
```

---

# Ejecución del proyecto

## Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecutar recolección de datos

```bash
python main.py
```

---

## Ejecutar entrenamiento del modelo

```bash
python modelotarea1.py
```

---

## Ejecutar procesamiento paralelo MPI

```bash
mpiexec -n 4 python data_engine_mpi.py
```

---

## Ejecutar dashboard

```bash
python appdash.py
```

---
