import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
from pathlib import Path
import json
from datetime import datetime


def cargar_y_preparar_datos(ruta_datos: str = "data/parquet/application_.parquet",
             tamano_test: float = 0.2, semilla_aleatoria: int = 42):
  print(" Cargando datos...")
  df = pd.read_parquet(ruta_datos)
  
  if 'TARGET' not in df.columns:
    raise ValueError("La columna TARGET no existe en los datos")
  
  X = df.drop(columns=['TARGET'])
  y = df['TARGET']
  
  columnas_numericas = X.select_dtypes(include=[np.number]).columns
  X = X[columnas_numericas]
  
  X = X.fillna(X.median())
  
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=tamano_test, random_state=semilla_aleatoria, stratify=y
  )
  
  print(f"Datos preparados:")
  print(f"Train: {X_train.shape}")
  print(f"Test: {X_test.shape}")
  print(f"Features: {X_train.shape[1]}")
  print(f"Target distribución (train): {y_train.value_counts(normalize=True).to_dict()}")
  
  return X_train, X_test, y_train, y_test


def entrenar_modelo(model, X_train, y_train, nombre_modelo: str):
  print(f"\n Entrenando {nombre_modelo}...")
  tiempo_inicio = datetime.now()
  
  model.fit(X_train, y_train)
  
  tiempo_entrenamiento = (datetime.now() - tiempo_inicio).total_seconds()
  print(f"{nombre_modelo} entrenado en {tiempo_entrenamiento:.2f} segundos")
  
  return model


def guardar_modelo(model, nombre_modelo: str, metrics: dict = None,
        directorio_artefactos: str = "artifacts"):
  ruta_artefactos = Path(directorio_artefactos)
  ruta_artefactos.mkdir(parents=True, exist_ok=True)
  
  ruta_modelo = ruta_artefactos / f"{nombre_modelo}.joblib"
  joblib.dump(model, ruta_modelo)
  print(f"Modelo guardado en {ruta_modelo}")
  
  if metrics:
    metadata = {
      "nombre_modelo": nombre_modelo,
      "timestamp": datetime.now().isoformat(),
      "metrics": metrics
    }
    
    ruta_metadatos = ruta_artefactos / f"{nombre_modelo}_metadata.json"
    with open(ruta_metadatos, 'w') as f:
      json.dump(metadata, f, indent=2)
    print(f"Metadatos guardados en {ruta_metadatos}")


def entrenar_todos_modelos(X_train, y_train):
  models = {
    'logistic_regression': LogisticRegression(
      max_iter=1000,
      random_state=42,
      class_weight='balanced'
    ),
    'xgboost': XGBClassifier(
      n_estimators=100,
      max_depth=6,
      learning_rate=0.1,
      random_state=42,
      scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
      use_label_encoder=False,
      eval_metric='logloss'
    )
  }
  
  modelos_entrenados = {}
  
  for name, model in models.items():
    trained_model = entrenar_modelo(model, X_train, y_train, name)
    modelos_entrenados[name] = trained_model
    guardar_modelo(trained_model, name)
  
  return modelos_entrenados


if __name__ == "__main__":
  print("=" * 70)
  print("ENTRENAMIENTO DE MODELOS")
  print("=" * 70)
  
  ruta_datos = Path("data/parquet/application_.parquet")
  
  if not ruta_datos.exists():
    print(f"\n Archivo no encontrado: {ruta_datos}")
    print("Por favor, coloca los datos en la carpeta data/parquet/")
  else:
    X_train, X_test, y_train, y_test = cargar_y_preparar_datos()
    print("\n" + "=" * 70)
    print("ENTRENANDO MODELOS")
    print("=" * 70)
    
    modelos_entrenados = entrenar_todos_modelos(X_train, y_train)
    
    print("\n" + "=" * 70)
    print(f"{len(modelos_entrenados)} modelos entrenados y guardados")
    print("=" * 70)
    
    test_data = {
      'X_test': X_test,
      'y_test': y_test
    }
    joblib.dump(test_data, 'artifacts/test_data.joblib')
    print("\n Datos de test guardados en artifacts/test_data.joblib")
