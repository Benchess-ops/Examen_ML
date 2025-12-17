import pandas as pd
import numpy as np
from pathlib import Path


class IngenieroCaracteristicas:
  def __init__(self):
    self.caracteristicas_creadas = []
  
  def crear_caracteristicas(self, df: pd.DataFrame) -> pd.DataFrame:
    df_caracteristicas = df.copy()
    
    print(" Creando nuevas features...")
    
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
      df_caracteristicas['CREDIT_INCOME_RATIO'] = (
        df_caracteristicas['AMT_CREDIT'] / (df_caracteristicas['AMT_INCOME_TOTAL'] + 1)
      )
      self.caracteristicas_creadas.append('CREDIT_INCOME_RATIO')
    
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
      df_caracteristicas['ANNUITY_INCOME_RATIO'] = (
        df_caracteristicas['AMT_ANNUITY'] / (df_caracteristicas['AMT_INCOME_TOTAL'] + 1)
      )
      self.caracteristicas_creadas.append('ANNUITY_INCOME_RATIO')
    
    if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
      df_caracteristicas['PAYMENT_RATE'] = (
        df_caracteristicas['AMT_ANNUITY'] / (df_caracteristicas['AMT_CREDIT'] + 1)
      )
      self.caracteristicas_creadas.append('PAYMENT_RATE')
    
    if 'DAYS_BIRTH' in df.columns:
      df_caracteristicas['AGE_YEARS'] = -df_caracteristicas['DAYS_BIRTH'] / 365
      self.caracteristicas_creadas.append('AGE_YEARS')
    
    if 'DAYS_EMPLOYED' in df.columns:
      df_caracteristicas['EMPLOYMENT_YEARS'] = df_caracteristicas['DAYS_EMPLOYED'].apply(
        lambda x: -x / 365 if x < 0 else np.nan
      )
      self.caracteristicas_creadas.append('EMPLOYMENT_YEARS')
    
    if 'EMPLOYMENT_YEARS' in df_caracteristicas.columns and 'AGE_YEARS' in df_caracteristicas.columns:
      df_caracteristicas['EMPLOYMENT_AGE_RATIO'] = (
        df_caracteristicas['EMPLOYMENT_YEARS'] / (df_caracteristicas['AGE_YEARS'] + 1)
      )
      self.caracteristicas_creadas.append('EMPLOYMENT_AGE_RATIO')
    
    if 'CNT_FAM_MEMBERS' in df.columns and 'CNT_CHILDREN' in df.columns:
      df_caracteristicas['ADULTS_IN_FAMILY'] = (
        df_caracteristicas['CNT_FAM_MEMBERS'] - df_caracteristicas['CNT_CHILDREN']
      )
      self.caracteristicas_creadas.append('ADULTS_IN_FAMILY')
    
    if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
      df_caracteristicas['INCOME_PER_CAPITA'] = (
        df_caracteristicas['AMT_INCOME_TOTAL'] / (df_caracteristicas['CNT_FAM_MEMBERS'] + 1)
      )
      self.caracteristicas_creadas.append('INCOME_PER_CAPITA')
    
    if 'AMT_GOODS_PRICE' in df.columns and 'AMT_CREDIT' in df.columns:
      df_caracteristicas['GOODS_PRICE_RATIO'] = (
        df_caracteristicas['AMT_GOODS_PRICE'] / (df_caracteristicas['AMT_CREDIT'] + 1)
      )
      self.caracteristicas_creadas.append('GOODS_PRICE_RATIO')
    
    if 'DAYS_LAST_PHONE_CHANGE' in df.columns:
      df_caracteristicas['YEARS_LAST_PHONE_CHANGE'] = (
        -df_caracteristicas['DAYS_LAST_PHONE_CHANGE'] / 365
      )
      self.caracteristicas_creadas.append('YEARS_LAST_PHONE_CHANGE')
    
    print(f"{len(self.caracteristicas_creadas)} nuevas features creadas:")
    for feat in self.caracteristicas_creadas:
      print(f"- {feat}")
    
    return df_caracteristicas
  
  def remover_valores_atipicos(self, df: pd.DataFrame, columns: list = None, 
            threshold: float = 3.0) -> pd.DataFrame:
    df_clean = df.copy()
    
    if columns is None:
      columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    initial_rows = len(df_clean)
    
    for col in columns:
      if col in df_clean.columns:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        df_clean = df_clean[z_scores < threshold]
    
    removed_rows = initial_rows - len(df_clean)
    print(f"Outliers removidos: {removed_rows} filas ({removed_rows/initial_rows*100:.2f}%)")
    
    return df_clean
  
  def select_top_features(self, df: pd.DataFrame, target_col: str = 'TARGET', 
              top_k: int = 50) -> list:
    if target_col not in df.columns:
      print(f"Columna {target_col} no encontrada")
      return []
    
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    columnas_numericas = [col for col in columnas_numericas if col != target_col]
    
    correlations = df[columnas_numericas].corrwith(df[target_col]).abs().sort_values(ascending=False)
    
    top_features = correlations.head(top_k).index.tolist()
    
    print(f"Top {top_k} features seleccionadas")
    print(f"Correlación máxima: {correlations.iloc[0]:.4f}")
    print(f"Correlación mínima (top {top_k}): {correlations.iloc[top_k-1]:.4f}")
    
    return top_features


if __name__ == "__main__":
  print("=" * 70)
  print("FEATURE ENGINEERING")
  print("=" * 70)
  
  ruta_datos = Path("data/parquet/application_.parquet")
  
  if not ruta_datos.exists():
    print(f"Archivo no encontrado: {ruta_datos}")
    print("Por favor, coloca los datos en la carpeta data/parquet/")
  else:
    df = pd.read_parquet(ruta_datos)
    print(f"\n Datos cargados: {df.shape}")
    
    fe = IngenieroCaracteristicas()
    
    df_engineered = fe.crear_caracteristicas(df)
    print(f"\n Datos con nuevas features: {df_engineered.shape}")
    
    if 'TARGET' in df_engineered.columns:
      top_features = fe.select_top_features(df_engineered, top_k=30)
      print(f"\n Top features:\n{', '.join(top_features[:10])}...")
