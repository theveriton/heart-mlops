# Heart Disease MLops Project

This project implements an end-to-end MLOps pipeline for predicting heart disease risk using the UCI Heart Disease dataset.

MLOps Assignment (Group 33)

## Contributors

- ARYAMANN SINGH - 2024aa05025
- ANANTHAN P   - 2024aa05692
- BALAJI R  - 2024aa05844
- BALSURE ANIKET K  - 2024aa05296
- SAURAV BANSAL - 2023aa05710

## Architecture

- **Data**: UCI Heart Disease dataset (14 features, binary target)
- **Model**: Tuned Logistic Regression or Random Forest with preprocessing pipeline
- **API**: FastAPI serving predictions via /predict endpoint
- **Tracking**: MLflow for experiments
- **CI/CD**: GitHub Actions for testing, linting, training
- **Containerization**: Docker
- **Deployment**: Kubernetes (local via manifests)
- **Monitoring**: Basic request logging and metrics endpoint

## Quick Start

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Download data and train:

```bash
python src/data_prep.py
python src/train.py
```

3. Run API locally:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Test with:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}}'
```

4. Build Docker image:

```bash
docker build -t heart-mlops:local .
docker run -p 8000:80 heart-mlops:local
```

5. Deploy to local Kubernetes (requires minikube or Docker Desktop):

```bash
kubectl apply -f k8s/
kubectl get services  # Note the external IP/port
```

## EDA

Run `notebooks/eda.ipynb` for exploratory data analysis including histograms, correlation heatmap, and class balance plots.

## Experiment Tracking

View MLflow experiments:

```bash
mlflow ui
```

## CI/CD

GitHub Actions workflow runs on push/PR: linting (flake8), tests (pytest), data prep, training, artifact upload.

## Monitoring

- API logs requests to console
- `/metrics` endpoint provides Prometheus-formatted metrics (request count)

### Local Prometheus/Grafana Setup

1. Start the API locally on port 8000 (as above).

2. Run Prometheus and Grafana:

```bash
docker-compose up -d
```

3. Access:
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

4. In Grafana:
   - Add Prometheus as data source (URL: http://prometheus:9090)
   - Create dashboard with query: `predict_requests_total`

5. Stop:

```bash
docker-compose down
```

## Project Structure

```
.
├── api/                 # FastAPI app
├── data/raw/           # Dataset
├── k8s/                # Kubernetes manifests
├── mlruns/             # MLflow experiments
├── notebooks/          # EDA notebook
├── screenshots/        # For reporting
├── src/                # Data prep and training
├── tests/              # Unit tests
├── Dockerfile          # Containerization
├── requirements.txt    # Dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Report

See `report.md` for the full assignment report including setup, EDA, modeling, CI/CD screenshots, and architecture diagram.
