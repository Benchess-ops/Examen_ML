from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from predict import PredictorRiesgoCredito


app = FastAPI(
  title="Home Credit Default Risk API",
  description="API para predecir el riesgo de incumplimiento de crédito",
  version="1.0.0"
)


class CreditApplication(BaseModel):
  features: Dict[str, Any] = Field(
    ...,
    description="Diccionario con las características del cliente",
    example={
      "AMT_CREDIT": 450000,
      "AMT_INCOME_TOTAL": 180000,
      "AMT_ANNUITY": 25000,
      "DAYS_BIRTH": -12000,
      "DAYS_EMPLOYED": -1500
    }
  )
  threshold: Optional[float] = Field(
    default=0.5,
    ge=0.0,
    le=1.0,
    description="Umbral de decisión (0-1)"
  )


class PredictionResponse(BaseModel):
  default_probability: float = Field(
    ...,
    description="Probabilidad de incumplimiento (0-1)"
  )
  prediction: int = Field(
    ...,
    description="Predicción binaria: 0 (no default) o 1 (default)"
  )
  decision: str = Field(
    ...,
    description="Decisión de negocio: APROBAR, REVISIÓN MANUAL o RECHAZAR"
  )
  risk_level: str = Field(
    ...,
    description="Nivel de riesgo: BAJO, MEDIO, ALTO, MUY ALTO"
  )
  threshold_used: float = Field(
    ...,
    description="Umbral utilizado para la decisión"
  )


predictor: Optional[PredictorRiesgoCredito] = None


@app.on_event("startup")
async def load_model():
  global predictor
  
  try:
    model_path = Path("artifacts/xgboost.joblib")
    
    if not model_path.exists():
      print(" Modelo no encontrado. Intentando cargar desde ruta alternativa...")
      model_path = Path(__file__).parent.parent / "artifacts" / "xgboost.joblib"
    
    predictor = PredictorRiesgoCredito(str(model_path))
    print(" Modelo cargado exitosamente")
    
  except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    predictor = None


@app.get("/")
async def root():
  """
  Endpoint raíz de la API.
  """
  return {
    "message": "Home Credit Default Risk API",
    "version": "1.0.0",
    "status": "running" if predictor else "model not loaded",
    "endpoints": {
      "health": "/health",
      "predict": "/predict (POST)",
      "docs": "/docs"
    }
  }


@app.get("/health")
async def health_check():
  if predictor is None:
    raise HTTPException(
      status_code=503,
      detail="Modelo no cargado. Por favor, entrena un modelo primero."
    )
  
  return {
    "status": "healthy",
    "model_loaded": True,
    "model_path": str(predictor.ruta_modelo)
  }


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: CreditApplication):
  if predictor is None:
    raise HTTPException(
      status_code=503,
      detail="Modelo no cargado. Por favor, entrena un modelo primero."
    )
  
  try:
    result = predictor.tomar_decision(
      features=application.features,
      threshold=application.threshold
    )
    
    return PredictionResponse(**result)
    
  except Exception as e:
    raise HTTPException(
      status_code=400,
      detail=f"Error en la predicción: {str(e)}"
    )


@app.post("/evaluate_risk", response_model=PredictionResponse)
async def evaluate_risk(application: CreditApplication):
  return await predict(application)


@app.get("/model/info")
async def model_info():
  if predictor is None:
    raise HTTPException(
      status_code=503,
      detail="Modelo no cargado"
    )
  
  info = {
    "model_path": str(predictor.ruta_modelo),
    "model_type": type(predictor.model).__name__,
    "feature_count": len(predictor.nombres_caracteristicas) if predictor.nombres_caracteristicas else "unknown"
  }
  
  return info


if __name__ == "__main__":
  print("=" * 70)
  print("INICIANDO API REST - HOME CREDIT DEFAULT RISK")
  print("=" * 70)
  print("\n📡 Servidor iniciando en http://localhost:8000")
  print("📚 Documentación disponible en http://localhost:8000/docs")
  print("\n💡 Endpoints disponibles:")
  print("  GET /       - Información de la API")
  print("  GET /health    - Estado de salud")
  print("  POST /predict   - Realizar predicción")
  print("  GET /model/info  - Información del modelo")
  print("\n" + "=" * 70)
  
  uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    log_level="info"
  )
