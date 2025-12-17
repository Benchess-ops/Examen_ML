# Examen : Predicción de Riesgo Crediticio -


# Nombre: Benjamin Antonio Riveros Silva - Giselle Alexandra Aguilera Gajardo
# Asignatura: Machine learning


## Descripción

El proyecto predice si un solicitante va a pagar o no su préstamo bancario. Se va a trabajar con 8 tablas diferentes que incluyen información de clientes, historial crediticio, pagos anteriores, datos de buró, etc.El dataset principal tiene 307,511 registros con 122 columnas. La estructura del proyecto sigue la metodología CRISP-DM organizada en carpetas de 01 a 05.

## Estructura del Proyecto

01_data_understanding/    - Carga de datos y análisis exploratorio
02_data_preparation/      - Limpieza, preprocesamiento y feature engineering  
03_modeling/              - Entrenamiento de modelos
04_evaluation/            - Evaluación y validación cruzada
05_deployment/            - API REST con FastAPI
artifacts/                - Modelos entrenados (.joblib)
data/parquet/             - Datasets (no incluidos en el repo)
requirements.txt          - Dependencias del proyecto

## Instalación

1. Instalar dependencias:

pip install -r requirements.txt

2. Descargar los datos de Kaggle y colocarlos en `data/parquet/`

## Uso

### Cargar y explorar datos

python 01_data_understanding/load_data.py

### Entrenar los modelos

python 03_modeling/train.py

Esto entrena Logistic Regression y XGBoost, guardando los modelos en `artifacts/`

### Levantar la API

python 05_deployment/app.py

La API queda disponible en `http://localhost:8000`

## API

### Endpoint: `/evaluate_risk`

Evalúa el riesgo de incumplimiento de un solicitante.

**Request:**

POST http://localhost:8000/evaluate_risk
Content-Type: application/json

```json
{
  "features": {
    "AMT_CREDIT": 450000,
    "AMT_INCOME_TOTAL": 180000,
    "AMT_ANNUITY": 25000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -1500
  },
  "threshold": 0.5
}
```

**Response:**
```json
{
  "default_probability": 0.3421,
  "prediction": 0,
  "decision": "REVISIÓN MANUAL",
  "risk_level": "MEDIO",
  "threshold_used": 0.5
}
```

### Decisiones

- **APROBAR**: Probabilidad < 30% (Riesgo BAJO)
- **REVISIÓN MANUAL**: Probabilidad 30-50% (Riesgo MEDIO)
- **RECHAZAR**: Probabilidad > 50% (Riesgo ALTO)

### Otros Endpoints

- `GET /` - Información de la API
- `GET /health` - Estado de salud del servicio
- `GET /docs` - Documentación interactiva (Swagger UI)

## Modelos

Se probaron 2 modelos:
- Logistic Regression (con class_weight='balanced')
- XGBoost (modelo final)

### Resultados del modelo XGBoost

- ROC AUC: 0.7511
- Accuracy: 70.98%
- Recall: 65.60%

El modelo prioriza el recall para minimizar falsos negativos (aprobar clientes que no pagarán)

## Ingeniería de características

Se crearon 10 features nuevas:
- CREDIT_INCOME_RATIO - Ratio crédito/ingreso
- ANNUITY_INCOME_RATIO - Ratio anualidad/ingreso
- PAYMENT_RATE - Tasa de pago
- AGE_YEARS - Edad en años
- EMPLOYMENT_YEARS - Años de empleo
- EMPLOYMENT_AGE_RATIO - Ratio empleo/edad
- ADULTS_IN_FAMILY - Adultos en familia
- INCOME_PER_CAPITA - Ingreso per cápita
- GOODS_PRICE_RATIO - Ratio precio bienes/crédito
- YEARS_LAST_PHONE_CHANGE - Años desde cambio de teléfono

## Manejo del Desbalance

El dataset está desbalanceado (92% pagan, 8% no pagan). Se aplicó:
- `class_weight='balanced'` en Logistic Regression
- `scale_pos_weight` en XGBoost

## Notas

- Los datos no están incluidos en el repositorio (son aproximadamente 500MB)
- Descargar desde Kaggle: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- Los modelos se guardan en `artifacts/` después del entrenamiento
- Se requiere Python 3.8+

## Datasets

| Archivo                       | Registros | Columnas  |
|-------------------------------|-----------|-----------|
| application_.parquet          | 307,511   | 122       |
| bureau.parquet                | 1,716,428 | 17        |
| bureau_balance.parquet        | 27,299,925| 3         |
| previous_application.parquet  | 1,670,214 | 37        |
| POS_CASH_balance.parquet      | 10,001,358| 8         |
| credit_card_balance.parquet   | 3,840,312 | 23        |
| installments_payments.parquet | 13,605,401| 8         |

## Tecnologías

- Python 3.8+
- Pandas / NumPy  
- Scikit-learn
- XGBoost
- FastAPI
- Joblib
- PyArrow


# Puede ocurrir un error el ejecutar los archivos py, hay que ejecutarlo 2 veces más, para que funcionen correctamente.
# Se tiene que crear una carpeta llamada data y una sub carpeta llamada parquet para subir los archivos