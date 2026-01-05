# Heart Disease MLops Project

Quick start:

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
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

4. Build Docker image:

```bash
docker build -t heart-mlops:local .
docker run -p 8000:80 heart-mlops:local
```

CI: GitHub Actions workflow is at `.github/workflows/ci.yml` and runs tests, data prep, and training.

Kubernetes manifests are in `k8s/` for local cluster deployment (replace image with your registry image).

See `report.md` for the assignment report and screenshots folder for reporting.
