import pandas as pd
import numpy as np
from sklearn.metrics import (
 roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
 confusion_matrix, classification_report, roc_curve, auc
)
import joblib
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def load_model_and_test_data(nombre_modelo: str = "xgboost",
      directorio_artefactos: str = "artifacts"):
 ruta_artefactos = Path(directorio_artefactos)
 
 ruta_modelo = ruta_artefactos / f"{nombre_modelo}.joblib"
 if not ruta_modelo.exists():
  raise FileNotFoundError(f"Modelo {nombre_modelo} no encontrado en {directorio_artefactos}")
 
 model = joblib.load(ruta_modelo)
 print(f"Modelo {nombre_modelo} cargado")
 
 test_data_path = ruta_artefactos / "test_data.joblib"
 if not test_data_path.exists():
  raise FileNotFoundError(f"Datos de test no encontrados en {directorio_artefactos}")
 
 test_data = joblib.load(test_data_path)
 X_test = test_data['X_test']
 y_test = test_data['y_test']
 
 print(f"Datos de test cargados: {X_test.shape}")
 
 return model, X_test, y_test


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
 metrics = {
  'accuracy': accuracy_score(y_true, y_pred),
  'precision': precision_score(y_true, y_pred, zero_division=0),
  'recall': recall_score(y_true, y_pred, zero_division=0),
  'f1_score': f1_score(y_true, y_pred, zero_division=0)
 }
 
 if y_pred_proba is not None:
  metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
 
 return metrics


def evaluar_modelo(model, X_test, y_test, nombre_modelo: str = "model"):
 print(f"\n Evaluando {nombre_modelo}...")
 
 y_pred = model.predict(X_test)
 
 try:
  y_pred_proba = model.predecir_probabilidad(X_test)[:, 1]
 except:
  y_pred_proba = None
 
 metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
 
 cm = confusion_matrix(y_test, y_pred)
 
 class_report = classification_report(y_test, y_pred, output_dict=True)
 
 results = {
  'nombre_modelo': nombre_modelo,
  'timestamp': datetime.now().isoformat(),
  'metrics': metrics,
  'confusion_matrix': cm.tolist(),
  'classification_report': class_report
 }
 
 return results, y_pred, y_pred_proba


def print_evaluation_results(results: dict):
 print("\n" + "=" * 70)
 print(f"RESULTADOS DE EVALUACIÓN: {results['nombre_modelo']}")
 print("=" * 60)
 
 print("\n MÉTRICAS PRINCIPALES:")
 metrics = results['metrics']
 for metric_name, value in metrics.items():
  print(f"{metric_name.upper()}: {value:.4f}")
 
 print("\n MATRIZ DE CONFUSIÓN:")
 cm = np.array(results['confusion_matrix'])
 print(f"TN: {cm[0][0]:>6} | FP: {cm[0][1]:>6}")
 print(f"FN: {cm[1][0]:>6} | TP: {cm[1][1]:>6}")
 
 print("\n REPORTE DE CLASIFICACIÓN:")
 class_report = results['classification_report']
 for label in ['0', '1']:
  if label in class_report:
   print(f"\n Clase {label}:")
   print(f"Precision: {class_report[label]['precision']:.4f}")
   print(f"Recall: {class_report[label]['recall']:.4f}")
   print(f"F1-Score: {class_report[label]['f1-score']:.4f}")


def save_evaluation_results(results: dict, output_path: str = "artifacts/evaluation_results.json"):
 Path(output_path).parent.mkdir(parents=True, exist_ok=True)
 
 with open(output_path, 'w') as f:
  json.dump(results, f, indent=2)
 
 print(f"\n Resultados guardados en {output_path}")


def graficar_matriz_confusion(cm, nombre_modelo: str = "model", 
       save_path: str = "artifacts/confusion_matrix.png"):
 plt.figure(figsize=(8, 6))
 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
    xticklabels=['No Default', 'Default'],
    yticklabels=['No Default', 'Default'])
 plt.title(f'Matriz de Confusión - {nombre_modelo}')
 plt.ylabel('Valor Real')
 plt.xlabel('Valor Predicho')
 plt.tight_layout()
 plt.savefig(save_path, dpi=300, bbox_inches='tight')
 plt.close()
 
 print(f"Matriz de confusión guardada en {save_path}")


def graficar_curva_roc(y_test, y_pred_proba, nombre_modelo: str = "model",
     save_path: str = "artifacts/roc_curve.png"):
 if y_pred_proba is None:
  print(" No hay probabilidades para generar curva ROC")
  return
 
 fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
 roc_auc = auc(fpr, tpr)
 
 plt.figure(figsize=(10, 7))
 plt.plot(fpr, tpr, color='darkorange', lw=2, 
    label=f'ROC curve (AUC = {roc_auc:.4f})')
 plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
    label='Random Classifier')
 plt.xlim([0.0, 1.0])
 plt.ylim([0.0, 1.05])
 plt.xlabel('False Positive Rate')
 plt.ylabel('True Positive Rate')
 plt.title(f'Receiver Operating Characteristic (ROC) - {nombre_modelo}')
 plt.legend(loc="lower right")
 plt.grid(alpha=0.3)
 plt.tight_layout()
 plt.savefig(save_path, dpi=300, bbox_inches='tight')
 plt.close()
 
 print(f"Curva ROC guardada en {save_path}")


if __name__ == "__main__":
 print("=" * 60)
 print("EVALUACIÓN DE MODELOS")
 print("=" * 60)
 
 nombre_modelo = "xgboost"
 
 try:
  model, X_test, y_test = load_model_and_test_data(nombre_modelo)
  
  results, y_pred, y_pred_proba = evaluar_modelo(model, X_test, y_test, nombre_modelo)
  
  print_evaluation_results(results)
  
  save_evaluation_results(results)
  
  print("\n" + "=" * 70)
  print("GENERANDO VISUALIZACIONES")
  print("=" * 60)
  
  cm = np.array(results['confusion_matrix'])
  graficar_matriz_confusion(cm, nombre_modelo)
  graficar_curva_roc(y_test, y_pred_proba, nombre_modelo)
  
  print("\n Evaluación completada")
  
 except FileNotFoundError as e:
  print(f"\n Error: {e}")
  print("Por favor, ejecuta primero train.py para entrenar los modelos")
