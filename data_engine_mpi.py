# data_engine_mpi.py
from mpi4py import MPI
import pandas as pd
import math
import time
import os


def procesar_datos():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Iniciar temporizador para tu reporte académico
    start_time = time.time()

    if rank == 0:
        print(f"[MPI_MASTER] Iniciando procesamiento con {size} nodos...")
        # Asegurarnos de que el directorio data exista
        os.makedirs("data", exist_ok=True)

        # 1. Cargar el dataset pesado
        try:
            df = pd.read_csv("data/vuelos_clima_1778363496.csv")
            print(f"[MPI_MASTER] Dataset original cargado: {len(df)} filas.")
        except FileNotFoundError:
            print("[MPI_MASTER] Error: No se encontró el dataset original.")
            comm.Abort()

        # 2. Dividir el dataset en N partes iguales usando PANDAS PURO (Evita perder nombres de columnas)
        chunk_size = math.ceil(len(df) / size)
        chunks = [df.iloc[i * chunk_size: (i + 1) * chunk_size] for i in range(size)]
    else:
        chunks = None

    # 3. SCATTER: Repartir los chunks a todos los procesos
    mi_chunk = comm.scatter(chunks, root=0)
    print(f"[Nodo {rank}] Recibió {len(mi_chunk)} filas para procesar.")

    # 4. PROCESAMIENTO PARALELO: Cada nodo calcula las métricas de su parte
    retrasos_locales = mi_chunk[mi_chunk["delayed"] > 0].groupby("dep_iata").agg(
        retraso_promedio=("delayed", "mean"),
        retraso_maximo=("delayed", "max"),
        vuelos_retrasados=("delayed", "count")
    ).reset_index()

    # 5. GATHER: El nodo maestro recolecta los resultados parciales
    todos_los_resultados = comm.gather(retrasos_locales, root=0)

    # 6. REDUCCIÓN: El maestro unifica y guarda la caché
    if rank == 0:
        print("[MPI_MASTER] Unificando resultados de todos los nodos...")
        # Concatenar todos los dataframes locales
        df_unificado = pd.concat(todos_los_resultados)

        # Agrupar de nuevo ya que varios nodos pueden tener los mismos aeropuertos
        df_final = df_unificado.groupby("dep_iata").agg(
            retraso_promedio=("retraso_promedio", "mean"),
            retraso_maximo=("retraso_maximo", "max"),
            vuelos_retrasados=("vuelos_retrasados", "sum")
        ).reset_index()

        # Ordenar por retraso para el Top 15 del Dashboard
        df_final = df_final.sort_values(by="retraso_promedio", ascending=False)

        # Guardar como caché ligero
        df_final.to_csv("data/mpi_cache_retrasos.csv", index=False)

        end_time = time.time()
        print(f"[MPI_MASTER] Caché generado en data/mpi_cache_retrasos.csv")
        print(f"[MPI_MASTER] Tiempo de ejecución MPI: {end_time - start_time:.4f} segundos")


if __name__ == "__main__":
    procesar_datos()