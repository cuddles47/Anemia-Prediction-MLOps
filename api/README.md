# Anemia Prediction Model API

This directory contains scripts for serving the best anemia prediction model via a REST API using MLflow.

## Components

- `get_best_model.py`: Script to retrieve the best model from MLflow based on a metric
- `serve_model.py`: Script to serve the model via MLflow's built-in REST API
- `test_api.py`: Script to test the model API with sample requests
- `Dockerfile`: Dockerfile for containerizing the model serving API

## How to Use

### Method 1: Running Locally

1. Make sure that MLflow server is running:
   ```
   cd /path/to/Anemia-Prediction-MLOps
   python deploy_mlflow.sh
   ```

2. Find and serve the best model:
   ```
   python -m api.serve_model --port 5001 --metric accuracy
   ```
   This will start the API server locally on port 5001.

3. In a new terminal, test the API:
   ```
   python -m api.test_api --port 5001 --samples 3
   ```

### Method 2: Using Docker Compose

1. Start all services using docker-compose:
   ```
   docker-compose up -d
   ```
   This will start:
   - MLflow tracking server (port 5000)
   - MinIO S3 storage (ports 9000, 9001)
   - MySQL database (port 3306)
   - Model serving API (port 5001)

2. Test the API (which will be available at http://localhost:5001/invocations):
   ```
   python -m api.test_api --port 5001 --samples 3
   ```

## API Usage

The API accepts POST requests with data in JSON format:

```
curl -X POST \
  http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "columns": ["age", "births_5_years", ...],
    "index": [0, 1, 2, ...],
    "data": [[27, 2, ...], ...]
  }'
```

The response will be a list of predicted values, with each value corresponding to the anemia level:

- 0: "Don't know"
- 1: "Moderate"
- 2: "Mild" 
- 3: "Not anemic"
- 4: "Severe"

## Production Deployment

For production deployment, consider:

1. Adding authentication to the API
2. Implementing HTTPS
3. Setting up health checks and monitoring
4. Configuring auto-scaling based on demand