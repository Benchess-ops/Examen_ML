import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import joblib
from pathlib import Path
import json


def evaluar_modelo_cv(model, X, y, cv: int = 5):
 print(f"Evaluando con {cv}-fold cross-validation...")
 
 roc_auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
 
 accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
 
 precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision', n_jobs=-1)
 
 recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall', n_jobs=-1)
 
 metrics = {
  'roc_auc_mean': roc_auc_scores.mean(),
  'roc_auc_std': roc_auc_scores.std(),
  'accuracy_mean': accuracy_scores.mean(),
  'accuracy_std': accuracy_scores.std(),
  'precision_mean': precision_scores.mean(),
  'precision_std': precision_scores.std(),
  'recall_mean': recall_scores.mean(),
  'recall_std': recall_scores.std()
 }
 
 return metrics


def comparar_modelos(models_dir: str = "artifacts", X_train=None, y_train=None):
 ruta_artefactos = Path(models_dir)
 
 if not ruta_artefactos.exists():
  print(f"Directorio {models_dir} no encontrado")
  return None

 model_files = list(ruta_artefactos.glob("*.joblib"))
 model_files = [f for f in model_files if f.stem not in ['test_data', 'preprocessor']]
 
 if not model_files:
  print(f"No se encontraron modelos en {models_dir}")
  return None
 
 print(f"Comparando {len(model_files)} modelos...")
 
 results = []
 
 for model_file in model_files:
  nombre_modelo = model_file.stem
  print(f"\n Evaluando {nombre_modelo}...")
  
  try:
   model = joblib.load(model_file)
  except Exception as e:
   print(f"Error al cargar {nombre_modelo}: {e}")
   continue
  
  if X_train is not None and y_train is not None:
   metrics = evaluar_modelo_cv(model, X_train, y_train)
   metrics['nombre_modelo'] = nombre_modelo
   results.append(metrics)
  else:
   results.append({'nombre_modelo': nombre_modelo})
 
 df_results = pd.DataFrame(results)
 
 if 'roc_auc_mean' in df_results.columns:
  df_results = df_results.sort_values('roc_auc_mean', ascending=False)
 
 return df_results


def seleccionar_mejor_modelo(comparison_df: pd.DataFrame, metric: str = 'roc_auc_mean'):
 if comparison_df is None or comparison_df.empty:
  print(" No hay modelos para comparar")
  return None
 
 if metric not in comparison_df.columns:
  print(f"Métrica {metric} no encontrada")
  return None
 
 mejor_modelo = comparison_df.iloc[0]['nombre_modelo']
 best_score = comparison_df.iloc[0][metric]
 
 print(f"\n Mejor modelo: {mejor_modelo}")
 print(f"{metric}: {best_score:.4f}")
 
 return mejor_modelo


def save_comparison_results(comparison_df: pd.DataFrame, 
       output_path: str = "artifacts/model_comparison.csv"):
 if comparison_df is None or comparison_df.empty:
  print(" No hay resultados para guardar")
  return
 
 comparison_df.to_csv(output_path, index=False)
 print(f"\n Resultados guardados en {output_path}")


if __name__ == "__main__":
 print("=" * 60)
 print("SELECCIÓN Y COMPARACIÓN DE MODELOS")
 print("=" * 60)
 
 ruta_artefactos = Path("artifacts")
 
 if not ruta_artefactos.exists() or not list(ruta_artefactos.glob("*.joblib")):
  print("\n No se encontraron modelos entrenados")
  print("Por favor, ejecuta primero train.py para entrenar los modelos")
 else:
  train_data_path = Path("data/parquet/application_.parquet")
  
  X_train, y_train = None, None
  
  if train_data_path.exists():
   print("\n Cargando datos de entrenamiento para validación cruzada...")
   df = pd.read_parquet(train_data_path)
   
   if 'TARGET' in df.columns:
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    
    columnas_numericas = X.select_dtypes(include=[np.number]).columns
    X = X[columnas_numericas].fillna(X[columnas_numericas].median())
    
    sample_size = min(10000, len(X))
    X_train = X.sample(n=sample_size, random_state=42)
    y_train = y.loc[X_train.index]
    
    print(f"Usando muestra de {sample_size} registros para CV")
  
  print("\n" + "=" * 70)
  print("COMPARACIÓN DE MODELOS")
  print("=" * 60)
  
  comparison_df = comparar_modelos(X_train=X_train, y_train=y_train)
  
  if comparison_df is not None:
   print("\n" + "=" * 70)
   print("RESULTADOS")
   print("=" * 60)
   print(comparison_df.to_string(index=False))
   
   if 'roc_auc_mean' in comparison_df.columns:
    mejor_modelo = seleccionar_mejor_modelo(comparison_df)
    
    save_comparison_results(comparison_df)
