# Proyecto Parcial 2 — Cómputo Paralelo y Distribuido

Proyecto desarrollado en Python para la implementación del manejo de cómputo en paralelo.  
Mediante el uso de peticiones a APIs se realiza un procesamiento de datos para alimentar un modelo de predicción, con la finalidad de calcular el tiempo de retraso que un avión puede llegar a tener. El modelo utilizado considera tanto datos de la aerolínea como factores meteorológicos. Además, se permite al usuario interactuar con el modelo mediante un dashboard interactivo, así como monitorear variables estadísticas.

---

# APIs utilizadas

##  AirLabs API

Información de vuelos:

 https://airlabs.co

---

## OpenWeatherMap API

Información climática:

 https://openweathermap.org/api

# Variables recolectadas

## Información de vuelos

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
