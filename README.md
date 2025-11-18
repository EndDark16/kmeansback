# K-Means Hospitals API

FastAPI backend that simula vecindarios dentro de una grilla m x m, ejecuta K-Means
para ubicar hospitales cercanos a los vecindarios, y expone el resultado vía REST.
Incluye dataset sintético, notebook de entrenamiento y Dockerfile listo para Render.

## Arquitectura

- **FastAPI** + `uvicorn` para servir JSON.
- **K-Means** implementado manualmente en `app/kmeans.py` para transparencia.
- Dataset sintético en `data/neighborhoods_synthetic.csv`.
- Notebook preparado para Colab en `notebooks/hospital_kmeans_training.ipynb`.
- Centroides preentrenados en `models/pretrained_hospitals.json`.

## Requisitos

- Python 3.11+
- Pip

## Instalación local

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

Arranca el servidor:

```bash
uvicorn app.main:app --reload --port 8000
```

Endpoints principales:

- `GET /health`
- `POST /kmeans/run` → body `{ "m": 20, "n": 100, "k": 4 }`
  - Respuesta incluye vecindarios, hospitales, asignaciones y métricas agregadas (`cluster_stats`, `overall_avg_distance`, `overall_max_distance`, `distance_bins`, `inertia`) listas para alimentar dashboards.
- `GET /kmeans/pretrained`

## Variables de entorno

| Variable | Descripción | Valor por defecto |
| --- | --- | --- |
| `CORS_ORIGINS` | Lista separada por comas para habilitar CORS. | `*` |
| `PRETRAINED_MODEL_PATH` | Ruta del JSON con centroides preentrenados. | `models/pretrained_hospitals.json` |

## Docker (Render)

```bash
docker build -t kmeans-hospitals .
docker run -it -p 8000:8000 kmeans-hospitals
```

Render: usar `uvicorn app.main:app --host 0.0.0.0 --port $PORT` en el comando de start.

## Notebook de entrenamiento

Ruta: `notebooks/hospital_kmeans_training.ipynb`.

Pasos:

1. Abrir el notebook en Colab.
2. Cargar `../data/neighborhoods_synthetic.csv` automáticamente o subirlo manualmente.
3. Analizar el dataset (histogramas, scatter plots, mapa de vecindarios).
4. Revisar la curva del codo para elegir `k`.
5. Ejecutar la celda final que guarda `pretrained_hospitals.json` y descargarlo.
6. Copiar el JSON a `models/pretrained_hospitals.json` en el backend antes de desplegar.

## Scripts auxiliares

- `scripts/generate_dataset.py` → regenera el CSV sintético.
- `scripts/create_pretrained_model.py` → corre K-Means sobre el dataset y actualiza el JSON.
- `scripts/create_notebook.py` → recrea el notebook si fuera necesario.

## Pruebas rápidas

```bash
python - <<\"PY\"
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
payload = {\"m\": 20, \"n\": 50, \"k\": 4}
print(client.post(\"/kmeans/run\", json=payload).json().keys())
print(client.get(\"/kmeans/pretrained\").json()[\"k\"])
PY
```

La API responde con vecindarios, hospitales y asignaciones listos para el frontend.
