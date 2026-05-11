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
