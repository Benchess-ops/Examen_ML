import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path


class Preprocesador:
  
  def __init__(self):
    self.imputador_numerico = SimpleImputer(strategy='median')
    self.imputador_categorico = SimpleImputer(strategy='most_frequent')
    self.escalador = StandardScaler()
    self.codificadores_etiquetas = {}
    self.columnas_numericas = []
    self.columnas_categoricas = []
  
  def ajustar(self, df: pd.DataFrame, columna_objetivo: str = 'TARGET') -> 'Preprocesador':
    X = df.drop(columns=[columna_objetivo]) if columna_objetivo in df.columns else df
    
    self.columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
    self.columnas_categoricas = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Features numéricas: {len(self.columnas_numericas)}")
    print(f"Features categóricas: {len(self.columnas_categoricas)}")
    
    if self.columnas_numericas:
      self.imputador_numerico.fit(X[self.columnas_numericas])
    
    if self.columnas_categoricas:
      self.imputador_categorico.fit(X[self.columnas_categoricas])
      
      for col in self.columnas_categoricas:
        codificador = LabelEncoder()
        datos_validos = X[col].dropna()
        codificador.fit(datos_validos)
        self.codificadores_etiquetas[col] = codificador
    
    if self.columnas_numericas:
      X_numerico_imputado = self.imputador_numerico.transform(X[self.columnas_numericas])
      self.escalador.fit(X_numerico_imputado)
    
    print(" Preprocesador ajustado correctamente")
    return self
  
  def transformar(self, df: pd.DataFrame, columna_objetivo: str = 'TARGET') -> pd.DataFrame:
    tiene_objetivo = columna_objetivo in df.columns
    if tiene_objetivo:
      y = df[columna_objetivo].copy()
      X = df.drop(columns=[columna_objetivo])
    else:
      X = df.copy()
    
    X_transformado = pd.DataFrame(index=X.index)
    
    if self.columnas_numericas:
      X_numerico_imputado = self.imputador_numerico.transform(X[self.columnas_numericas])
      X_numerico_escalado = self.escalador.transform(X_numerico_imputado)
      
      df_numerico = pd.DataFrame(
        X_numerico_escalado,
        columns=self.columnas_numericas,
        index=X.index
      )
      X_transformado = pd.concat([X_transformado, df_numerico], axis=1)
    
    if self.columnas_categoricas:
      X_categorico_imputado = self.imputador_categorico.transform(X[self.columnas_categoricas])
      
      categoricas_codificadas = []
      for i, col in enumerate(self.columnas_categoricas):
        codificador = self.codificadores_etiquetas[col]
        datos_col = X_categorico_imputado[:, i]
        
        codificado = np.zeros(len(datos_col), dtype=int)
        for j, val in enumerate(datos_col):
          if val in codificador.classes_:
            codificado[j] = codificador.transform([val])[0]
          else:
            codificado[j] = -1
        
        categoricas_codificadas.append(codificado)
      
      df_categorico = pd.DataFrame(
        np.array(categoricas_codificadas).T,
        columns=self.columnas_categoricas,
        index=X.index
      )
      X_transformado = pd.concat([X_transformado, df_categorico], axis=1)
    
    if tiene_objetivo:
      X_transformado[columna_objetivo] = y
    
    print(f"Datos transformados: {X_transformado.shape}")
    return X_transformado
  
  def ajustar_transformar(self, df: pd.DataFrame, columna_objetivo: str = 'TARGET') -> pd.DataFrame:
    self.ajustar(df, columna_objetivo)
    return self.transformar(df, columna_objetivo)
  
  def guardar(self, ruta_archivo: str = "artifacts/preprocesador.joblib") -> None:
    Path(ruta_archivo).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(self, ruta_archivo)
    print(f"Preprocesador guardado en {ruta_archivo}")
  
  @staticmethod
  def cargar(ruta_archivo: str = "artifacts/preprocesador.joblib") -> 'Preprocesador':
    preprocesador = joblib.load(ruta_archivo)
    print(f"Preprocesador cargado desde {ruta_archivo}")
    return preprocesador


if __name__ == "__main__":
  print("=" * 70)
  print("PREPROCESAMIENTO DE DATOS")
  print("=" * 70)
  
  ruta_datos = Path("data/parquet/application_.parquet")
  
  if not ruta_datos.exists():
    print(f"Archivo no encontrado: {ruta_datos}")
    print("Por favor, coloca los datos en la carpeta data/parquet/")
  else:
    df = pd.read_parquet(ruta_datos)
    print(f"\n Datos cargados: {df.shape}")
    
    preprocesador = Preprocesador()
    df_transformado = preprocesador.ajustar_transformar(df)
    
    print(f"\n Datos preprocesados: {df_transformado.shape}")
    print(f"\nPrimeras filas:\n{df_transformado.head()}")
    
    preprocesador.guardar()
