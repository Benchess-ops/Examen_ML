import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score
import joblib
from pathlib import Path
import json
from datetime import datetime


def perform_cross_validation(model, X, y, cv: int = 5, scoring: dict = None):

  if scoring is None:
    scoring = {
      'roc_auc': 'roc_auc',
      'accuracy': 'accuracy',
      'precision': make_scorer(precision_score, zero_division=0),
      'recall': make_scorer(recall_score, zero_division=0)
    }
  
  print(f"Ejecutando {cv}-fold cross-validation...")
  print(f"Métricas: {', '.join(scoring.keys())}")
  
  skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
  
  cv_results = cross_validate(
    model, X, y,
    cv=skf,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1,
    verbose=0
  )
  
  return cv_results


def summarize_cv_results(cv_results: dict, nombre_modelo: str = "model"):
  summary = {
    'nombre_modelo': nombre_modelo,
    'timestamp': datetime.now().isoformat(),
    'n_splits': len(cv_results['fit_time']),
    'metrics': {}
  }
  
  for key in cv_results.keys():
    if key.startswith('test_'):
      metric_name = key.replace('test_', '')
      values = cv_results[key]
      
      summary['metrics'][metric_name] = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'values': values.tolist()
      }
  
  summary['fit_time'] = {
    'mean': float(np.mean(cv_results['fit_time'])),
    'total': float(np.sum(cv_results['fit_time']))
  }
  
  return summary


def print_cv_summary(summary: dict):
  print("\n" + "=" * 70)
  print(f"VALIDACIÓN CRUZADA: {summary['nombre_modelo']}")
  print("=" * 70)
  
  print(f"\n Configuración: {summary['n_splits']} folds")
  print(f"Tiempo total de entrenamiento: {summary['fit_time']['total']:.2f} segundos")
  
  print("\n MÉTRICAS DE VALIDACIÓN CRUZADA:")
  
  for metric_name, stats in summary['metrics'].items():
    print(f"\n  {metric_name.upper()}:")
    print(f"Media: {stats['mean']:.4f} (± {stats['std']:.4f})")
    print(f"Rango: [{stats['min']:.4f}, {stats['max']:.4f}]")


def validate_all_models(X, y, models_dir: str = "artifacts", cv: int = 5):
  ruta_artefactos = Path(models_dir)
  
  if not ruta_artefactos.exists():
    print(f"Directorio {models_dir} no encontrado")
    return []
  
  model_files = list(ruta_artefactos.glob("*.joblib"))
  model_files = [f for f in model_files if f.stem not in ['test_data', 'preprocessor', 'preprocesador']]
  
  if not model_files:
    print(f"No se encontraron modelos en {models_dir}")
    return []
  
  print(f"Validando {len(model_files)} modelos...")
  
  all_results = []
  
  for model_file in model_files:
    nombre_modelo = model_file.stem
    print(f"\n{'='*70}")
    print(f"Validando: {nombre_modelo}")
    print('='*70)
    
    try:
      model = joblib.load(model_file)
    except Exception as e:
      print(f"Error al cargar {nombre_modelo}: {e}")
      continue
    
    cv_results = perform_cross_validation(model, X, y, cv=cv)
    
    summary = summarize_cv_results(cv_results, nombre_modelo)
    
    print_cv_summary(summary)
    
    all_results.append(summary)
  
  return all_results


def save_cv_results(results: list, output_path: str = "artifacts/cv_results.json"):
  Path(output_path).parent.mkdir(parents=True, exist_ok=True)
  
  with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
  
  print(f"\n Resultados de CV guardados en {output_path}")


def compare_cv_results(results: list):
  if not results:
    return None
  
  comparison_data = []
  
  for result in results:
    row = {'nombre_modelo': result['nombre_modelo']}
    
    for metric_name, stats in result['metrics'].items():
      row[f'{metric_name}_mean'] = stats['mean']
      row[f'{metric_name}_std'] = stats['std']
    
    comparison_data.append(row)
  
  df = pd.DataFrame(comparison_data)
  
  if 'roc_auc_mean' in df.columns:
    df = df.sort_values('roc_auc_mean', ascending=False)
  
  return df


if __name__ == "__main__":
  print("=" * 70)
  print("VALIDACIÓN CRUZADA DE MODELOS")
  print("=" * 70)
  
  ruta_datos = Path("data/parquet/application_.parquet")
  
  if not ruta_datos.exists():
    print(f"\n Archivo no encontrado: {ruta_datos}")
    print("Por favor, coloca los datos en la carpeta data/parquet/")
  else:
    print("\n Cargando datos...")
    df = pd.read_parquet(ruta_datos)
    
    if 'TARGET' not in df.columns:
      print(" La columna TARGET no existe en los datos")
    else:
      X = df.drop(columns=['TARGET'])
      y = df['TARGET']
      
      columnas_numericas = X.select_dtypes(include=[np.number]).columns
      X = X[columnas_numericas].fillna(X[columnas_numericas].median())
      
      sample_size = min(15000, len(X))
      print(f"Usando muestra de {sample_size:,} registros")
      
      X_sample = X.sample(n=sample_size, random_state=42)
      y_sample = y.loc[X_sample.index]
      cv_results = validate_all_models(X_sample, y_sample, cv=5)
      
      if cv_results:
        save_cv_results(cv_results)
        
        print("\n" + "=" * 70)
        print("COMPARACIÓN DE MODELOS (VALIDACIÓN CRUZADA)")
        print("=" * 70)
        
        comparison_df = compare_cv_results(cv_results)
        if comparison_df is not None:
          print("\n", comparison_df.to_string(index=False))
        
        print("\n Validación cruzada completada")
