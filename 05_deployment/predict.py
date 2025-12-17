import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Union, Dict, List


class PredictorRiesgoCredito:
  
  def __init__(self, ruta_modelo: str = "artifacts/xgboost.joblib"):
    self.ruta_modelo = Path(ruta_modelo)
    self.model = None
    self.nombres_caracteristicas = None
    self.cargar_modelo()
  
  def cargar_modelo(self):
    """
    Carga el modelo desde disco.
    """
    if not self.ruta_modelo.exists():
      raise FileNotFoundError(f"Modelo no encontrado en {self.ruta_modelo}")
    
    self.model = joblib.load(self.ruta_modelo)
    
    if hasattr(self.model, 'feature_names_in_'):
      self.nombres_caracteristicas = self.model.feature_names_in_.tolist()
    
    print(f"Modelo cargado desde {self.ruta_modelo}")
  
  def validar_entrada(self, features: Dict) -> pd.DataFrame:
    if isinstance(features, dict):
      df = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame):
      df = features.copy()
    else:
      raise ValueError("Input debe ser dict o DataFrame")
    
    if self.nombres_caracteristicas:
      caracteristicas_faltantes = set(self.nombres_caracteristicas) - set(df.columns)
      if caracteristicas_faltantes:
        missing_df = pd.DataFrame(0, index=df.index, columns=list(caracteristicas_faltantes))
        df = pd.concat([df, missing_df], axis=1)
      
      df = df[self.nombres_caracteristicas]
    
    df = df.fillna(0)
    
    return df
  
  def predecir_probabilidad(self, features: Union[Dict, pd.DataFrame]) -> float:
    df = self.validar_entrada(features)
    proba = self.model.predict_proba(df)[:, 1][0]
    
    return float(proba)
  
  def predict(self, features: Union[Dict, pd.DataFrame], 
        threshold: float = 0.5) -> int:
    proba = self.predecir_probabilidad(features)
    return int(proba >= threshold)
  
  def tomar_decision(self, features: Union[Dict, pd.DataFrame],
           threshold: float = 0.5) -> Dict:
    proba = self.predecir_probabilidad(features)
    prediction = int(proba >= threshold)
    if proba < 0.3:
      decision = "APROBAR"
      nivel_riesgo = "BAJO"
    elif proba < 0.5:
      decision = "REVISIÓN MANUAL"
      nivel_riesgo = "MEDIO"
    elif proba < 0.7:
      decision = "RECHAZAR"
      nivel_riesgo = "ALTO"
    else:
      decision = "RECHAZAR"
      nivel_riesgo = "MUY ALTO"
    
    return {
      "default_probability": round(proba, 4),
      "prediction": prediction,
      "decision": decision,
      "risk_level": nivel_riesgo,
      "threshold_used": threshold
    }
  
  def predecir_lote(self, lista_caracteristicas: List[Dict],
           threshold: float = 0.5) -> List[Dict]:
    results = []
    
    for features in lista_caracteristicas:
      try:
        result = self.tomar_decision(features, threshold)
        result['status'] = 'success'
      except Exception as e:
        result = {
          'status': 'error',
          'error_message': str(e)
        }
      
      results.append(result)
    
    return results


def load_predictor(nombre_modelo: str = "xgboost") -> PredictorRiesgoCredito:
  ruta_modelo = f"artifacts/{nombre_modelo}.joblib"
  return PredictorRiesgoCredito(ruta_modelo)


if __name__ == "__main__":
  print("=" * 70)
  print("MÓDULO DE PREDICCIÓN")
  print("=" * 70)
  
  try:
    ruta = Path(__file__).parent.parent / "artifacts" / "xgboost.joblib"
    predictor = PredictorRiesgoCredito(str(ruta))
    print("\n Ejemplo de predicción:")
    
    example_features = {
      'AMT_CREDIT': 450000,
      'AMT_INCOME_TOTAL': 180000,
      'AMT_ANNUITY': 25000,
      'DAYS_BIRTH': -12000,
      'DAYS_EMPLOYED': -1500
    }
    
    print(f"\nFeatures de entrada: {example_features}")
    result = predictor.tomar_decision(example_features)
    
    print("\n Resultado de la predicción:")
    print(f"Probabilidad de incumplimiento: {result['default_probability']:.2%}")
    print(f"Predicción: {result['prediction']} ({'Default' if result['prediction'] == 1 else 'No Default'})")
    print(f"Decisión: {result['decision']}")
    print(f"Nivel de riesgo: {result['risk_level']}")
    
    print("\n Módulo de predicción funcionando correctamente")
    
  except FileNotFoundError as e:
    print(f"\n Error: {e}")
    print("Por favor, entrena un modelo primero ejecutando 03_modeling/train.py")
