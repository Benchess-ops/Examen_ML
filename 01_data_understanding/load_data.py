import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq


def cargar_datos_parquet(ruta_datos: str = "data/parquet", 
                         archivos_excluir: list = None,
                         solo_metadata: bool = False) -> dict:
  directorio_datos = Path(ruta_datos)
  
  if not directorio_datos.exists():
    raise FileNotFoundError(f"El directorio {ruta_datos} no existe")
  
  dataframes = {}
  archivos_parquet = list(directorio_datos.glob("*.parquet"))
  
  if archivos_excluir is None:
    archivos_excluir = []
  
  if not archivos_parquet:
    print(f"No se encontraron archivos Parquet en {ruta_datos}")
    return dataframes
  
  print(f"Cargando datos desde {ruta_datos}...")
  
  for ruta_archivo in archivos_parquet:
    nombre_archivo = ruta_archivo.stem
    
    if nombre_archivo in archivos_excluir:
      print(f"Saltando {nombre_archivo}.parquet (excluido)")
      continue
    
    print(f"Cargando {nombre_archivo}.parquet...")
    
    try:
      if solo_metadata:
        parquet_file = pq.ParquetFile(ruta_archivo)
        metadata = parquet_file.metadata
        print(f"Shape: ({metadata.num_rows}, {metadata.num_columns})")
        dataframes[nombre_archivo] = {"metadata": metadata, "path": ruta_archivo}
      else:
        df = pd.read_parquet(ruta_archivo, engine='pyarrow')
        dataframes[nombre_archivo] = df
        print(f"Shape: {df.shape}")
    except Exception as e:
      print(f"Error al cargar {nombre_archivo}: {str(e)}")
  
  print(f"Total de archivos cargados: {len(dataframes)}\n")
  
  return dataframes


def cargar_archivo_parquet(nombre_archivo: str, 
                           ruta_datos: str = "data/parquet",
                           columnas: list = None) -> pd.DataFrame:
  if not nombre_archivo.endswith('.parquet'):
    nombre_archivo = f"{nombre_archivo}.parquet"
  
  ruta = Path(ruta_datos) / nombre_archivo
  
  if not ruta.exists():
    raise FileNotFoundError(f"Archivo no encontrado: {ruta}")
  
  print(f"Cargando {nombre_archivo}...")
  df = pd.read_parquet(ruta, columns=columnas, engine='pyarrow')
  print(f"Shape: {df.shape}")
  return df


def obtener_resumen_datos(dataframes: dict) -> None:
  """Muestra un resumen de los DataFrames cargados."""
  if not dataframes:
    print("No hay datos para resumir.")
    return
  
  print("=" * 60)
  print("RESUMEN DE DATOS CARGADOS")
  print("=" * 60)
  
  for nombre, df in dataframes.items():
    print(f"\n{nombre}")
    print(f"Shape: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Columnas: {', '.join(df.columns[:5].tolist())}" + 
       (f", ... (+{len(df.columns)-5} mas)" if len(df.columns) > 5 else ""))


if __name__ == "__main__":
  archivos_grandes = ["bureau_balance", "POS_CASH_balance", "installments_payments", "credit_card_balance"]
  
  print("=" * 60)
  print("CARGA INICIAL - Archivos principales")
  print("=" * 60)
  datos = cargar_datos_parquet(archivos_excluir=archivos_grandes)
  obtener_resumen_datos(datos)
  
  print("\n" + "=" * 60)
  print("METADATA - Archivos grandes (no cargados en memoria)")
  print("=" * 60)
  for archivo in archivos_grandes:
    try:
      ruta = Path("data/parquet") / f"{archivo}.parquet"
      if ruta.exists():
        pf = pq.ParquetFile(ruta)
        print(f"{archivo}: {pf.metadata.num_rows:,} filas x {pf.metadata.num_columns} columnas")
    except Exception as e:
      print(f"Error leyendo metadata de {archivo}: {e}")
  
  if "application_" in datos:
    df_entrenamiento = datos["application_"]
    print("\n" + "=" * 70)
    print("INFORMACION ADICIONAL: application_")
    print("=" * 60)
    print(f"\nPrimeras columnas:\n{df_entrenamiento.head()}")
    print(f"\nTipos de datos:\n{df_entrenamiento.dtypes.value_counts()}")
    
    if "TARGET" in df_entrenamiento.columns:
      print(f"\nDistribucion de TARGET:")
      print(df_entrenamiento["TARGET"].value_counts(normalize=True))
