import asyncio
import aiohttp
import csv
import time
import os

from dotenv import load_dotenv


load_dotenv()

AIRLABS_API_KEY = os.getenv("AIRLABS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not AIRLABS_API_KEY:
    raise ValueError("❌ No se encontro AIRLABS_API_KEY en .env")

if not OPENWEATHER_API_KEY:
    raise ValueError("❌ No se encontro OPENWEATHER_API_KEY en .env")


AIRLABS_URL = "https://airlabs.co/api/v9/schedules"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


semaforo = asyncio.Semaphore(10)


CSV_FILE = f"vuelos_clima_{int(time.time())}.csv"

#Códigos de aeropuerto para obtener información de airlabs y sus coordenadas para obtener el clima 

AEROPUERTOS_COORDS = {
    "ORD": (41.9742, -87.9073),
    "ATL": (33.6407, -84.4277),
    "DFW": (32.8998, -97.0403),
    "DEN": (39.8561, -104.6737),
    "CLT": (35.2144, -80.9473),
    "PVG": (31.1443, 121.8083),
    "CAN": (23.3924, 113.2988),
    "DEL": (28.5562, 77.1000),
    "HND": (35.5494, 139.7798),
    "PHX": (33.4342, -112.0116),
    "LAX": (33.9416, -118.4085),
    "IST": (41.2753, 28.7519),
    "AMS": (52.3105, 4.7683),
    "KUL": (2.7456, 101.7072),
    "SZX": (22.6393, 113.8107),
    "LHR": (51.4700, -0.4543),
    "PEK": (40.0799, 116.6031),
    "TFU": (30.5785, 103.9471),
    "ICN": (37.4602, 126.4407),
    "IAH": (29.9902, -95.3368),
    "CGK": (-6.1256, 106.6559),
    "CDG": (49.0097, 2.5479),
    "MAD": (40.4983, -3.5676),
    "MIA": (25.7959, -80.2870),
    "SEA": (47.4502, -122.3088),
    "BKK": (13.6900, 100.7501),
    "LAS": (36.0840, -115.1537),
    "JFK": (40.6413, -73.7781),
    "MCO": (28.4312, -81.3081),
    "LGA": (40.7769, -73.8740),
    "SIN": (1.3644, 103.9915),
    "PKX": (39.5098, 116.4105),
    "KMG": (25.1019, 102.9292),
    "FRA": (50.0379, 8.5622),
    "BOS": (42.3656, -71.0096),
    "CKG": (29.7192, 106.6416),
    "XIY": (34.4471, 108.7516),
    "EWR": (40.6895, -74.1745),
    "HKG": (22.3080, 113.9185),
    "BOM": (19.0896, 72.8656),
    "BOG": (4.7016, -74.1469),
    "HGH": (30.2361, 120.4355),
    "SFO": (37.6213, -122.3790),
    "FCO": (41.8003, 12.2389),
    "BCN": (41.2974, 2.0833),
    "DCA": (38.8512, -77.0402),
    "DTW": (42.2162, -83.3554),
    "MNL": (14.5086, 121.0198),
    "SHA": (31.1979, 121.3363),
    "BLR": (13.1986, 77.7066),
    "PHL": (39.8744, -75.2424),
    "MSP": (44.8848, -93.2223),
    "DXB": (25.2532, 55.3657),
    "YYZ": (43.6777, -79.6248),
    "SYD": (-33.9399, 151.1753),
    "MUC": (48.3538, 11.7861),
    "SGN": (10.8188, 106.6519),
    "MEX": (19.4361, -99.0719),
    "BNA": (36.1263, -86.6774),
    "SLC": (40.7899, -111.9791),
    "TPE": (25.0797, 121.2342),
    "ATH": (37.9364, 23.9475),
    "RUH": (24.9576, 46.6988),
    "IAD": (38.9531, -77.4565),
    "DUB": (53.4213, -6.2701),
    "NKG": (31.7420, 118.8620),
    "NRT": (35.7720, 140.3929),
    "ORY": (48.7262, 2.3652),
    "SAN": (32.7338, -117.1933),
    "WUH": (30.7838, 114.2081),
    "DMK": (13.9126, 100.6070),
    "CPH": (55.6181, 12.6560),
    "JED": (21.6702, 39.1525),
    "FLL": (26.0742, -80.1506),
    "VIE": (48.1103, 16.5697),
    "SVO": (55.9726, 37.4146),
    "CGO": (34.5197, 113.8410),
    "PMI": (39.5517, 2.7388),
    "ZRH": (47.4581, 8.5555),
    "MEL": (-37.6690, 144.8410),
    "CSX": (28.1892, 113.2206),
    "HAN": (21.2212, 105.8072),
    "HYD": (17.2403, 78.4294),
    "BWI": (39.1754, -76.6684),
    "LGW": (51.1537, -0.1821),
    "FUK": (33.5859, 130.4507),
    "CTU": (30.5785, 103.9471),
    "JNB": (-26.1337, 28.2420),
    "TAO": (36.2661, 120.3744),
    "XMN": (24.5440, 118.1277),
    "URC": (43.9071, 87.4742),
    "YVR": (49.1967, -123.1815),
    "CJU": (33.5113, 126.4928),
    "AUS": (30.1975, -97.6664),
    "SAW": (40.8986, 29.3092),
    "HAK": (19.9349, 110.4589),
    "OSL": (60.1939, 11.1004),
    "LIS": (38.7742, -9.1342),
    "BNE": (-27.3842, 153.1175),
    "AGP": (36.6749, -4.4991)
}

# Solicitud de peticiones del clima

async def obtener_clima(session, lat, lon):

    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    try:

        async with session.get(WEATHER_URL, params=params) as response:

            data = await response.json()

            return {
                "weather_temp": data.get("main", {}).get("temp"),
                "weather_temp_max": data.get("main", {}).get("temp_max"),
                "weather_temp_min": data.get("main", {}).get("temp_min"),
                "weather_pressure": data.get("main", {}).get("pressure"),
                "weather_humidity": data.get("main", {}).get("humidity"),
                "weather_sea_level": data.get("main", {}).get("sea_level"),
                "weather_visibility": data.get("visibility"),
                "weather_wind_speed": data.get("wind", {}).get("speed"),
                "weather_rain_1h": data.get("rain", {}).get("1h", 0),
                "weather_clouds": data.get("clouds", {}).get("all")
            }

    except Exception as e:

        print(f"Error clima: {e}")

        return {
            "weather_temp": None,
            "weather_temp_max": None,
            "weather_temp_min": None,
            "weather_pressure": None,
            "weather_humidity": None,
            "weather_sea_level": None,
            "weather_visibility": None,
            "weather_wind_speed": None,
            "weather_rain_1h": None,
            "weather_clouds": None
        }

#Solicitudes asincronas a ambas api dadas por el código del aeropuerto

async def obtener_vuelos(session, aeropuerto, lat, lon):

    async with semaforo:

        try:

            clima = await obtener_clima(session, lat, lon)

            params = {
                "dep_iata": aeropuerto,
                "api_key": AIRLABS_API_KEY
            }

            async with session.get(AIRLABS_URL, params=params) as response:

                if response.status != 200:

                    print(f"Error {response.status} en {aeropuerto}")
                    return []

                data = await response.json()

                vuelos = data.get("response", [])

                resultados = []

                for vuelo in vuelos:

                    fila = {

                        # Datos obtenidos de la api de los vuelos

                        "flight_iata": vuelo.get("flight_iata"),
                        "dep_iata": vuelo.get("dep_iata"),
                        "dep_time": vuelo.get("dep_time"),
                        "dep_estimated": vuelo.get("dep_estimated"),
                        "arr_iata": vuelo.get("arr_iata"),
                        "arr_time": vuelo.get("arr_time"),
                        "status": vuelo.get("status"),

                        # Datos obtenidos de la api del clima 

                        "weather_temp": clima["weather_temp"],
                        "weather_temp_max": clima["weather_temp_max"],
                        "weather_temp_min": clima["weather_temp_min"],
                        "weather_pressure": clima["weather_pressure"],
                        "weather_humidity": clima["weather_humidity"],
                        "weather_sea_level": clima["weather_sea_level"],
                        "weather_visibility": clima["weather_visibility"],
                        "weather_wind_speed": clima["weather_wind_speed"],
                        "weather_rain_1h": clima["weather_rain_1h"],
                        "weather_clouds": clima["weather_clouds"]
                    }

                    resultados.append(fila)

                print(f"{aeropuerto}: {len(resultados)} vuelos")

                await asyncio.sleep(0.2)

                return resultados

        except Exception as e:

            print(f"Error en {aeropuerto}: {e}")
            return []


def guardar_csv(datos):

    columnas = [

        #Columnas referentes al status de los vuelos

        "flight_iata",
        "dep_iata",
        "dep_time",
        "dep_estimated",
        "arr_iata",
        "arr_time",
        "status",

        #Columnas referentes al clima

        "weather_temp",
        "weather_temp_max",
        "weather_temp_min",
        "weather_pressure",
        "weather_humidity",
        "weather_sea_level",
        "weather_visibility",
        "weather_wind_speed",
        "weather_rain_1h",
        "weather_clouds"
    ]

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as archivo:

        writer = csv.DictWriter(
            archivo,
            fieldnames=columnas
        )

        writer.writeheader()
        writer.writerows(datos)

    print(f"\nCSV guardado: {CSV_FILE}")


async def main():

    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        tareas = []

        for aeropuerto, (lat, lon) in AEROPUERTOS_COORDS.items():

            tareas.append(
                obtener_vuelos(
                    session,
                    aeropuerto,
                    lat,
                    lon
                )
            )

        resultados = await asyncio.gather(*tareas)


        datos_finales = [
            vuelo
            for lista in resultados
            for vuelo in lista
        ]

        print(f"\nTotal vuelos: {len(datos_finales)}")

        guardar_csv(datos_finales)


if __name__ == "__main__":
    asyncio.run(main())