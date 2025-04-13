# MLflow Tracking Server Deployment

![MLflow Logo](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/_static/MLflow-logo-final-black.png)

## Overview

This repository provides a streamlined solution for deploying a production-ready MLflow tracking server using Docker Compose. The setup includes:

- **MLflow Server**: Tracking experiments, parameters, metrics, and artifacts
- **MinIO**: S3-compatible object storage for model artifacts
- **MySQL**: Relational database for metadata storage

Perfect for teams who need a self-hosted, persistent model registry and experiment tracking solution.

## Quick Start

```bash
# Clone and start the server
git clone https://github.com/cuddles47/Anemia-Prediction-MLOps
cd Anemia-Prediction-MLOps
./deploy_mlflow.sh
```

Once deployed, access:
- **MLflow UI**: http://localhost:5000
- **MinIO Console**: http://localhost:9000 (default credentials: minio/minio123)

## Setup Requirements

- Docker Engine (19.03.0+)
- Docker Compose (1.27.0+)
- 4GB+ RAM available for containers
- 10GB+ disk space

Need to install dependencies? Use our helper scripts:
```bash
# Install Docker
./install_docker.sh

# Install Anaconda (for local MLflow client)
./anaconda_install.sh
```

## Configuration

Create a `.env` file based on the `env.example` template:

```bash
cp env.example .env
```

Key configurations:
- `MYSQL_DATABASE`: Database name for MLflow metadata
- `MYSQL_USER`: Database username
- `MYSQL_PASSWORD`: Database password
- `AWS_ACCESS_KEY_ID`: MinIO access key (default: minio)
- `AWS_SECRET_ACCESS_KEY`: MinIO secret key (default: minio123)

## Client Configuration

Connect your local MLflow client by configuring these environment variables:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
```

## Project Structure

```
├── anaconda_install.sh    # Script to install Anaconda
├── deploy_mlflow.sh       # One-command deployment script
├── docker-compose.yml     # Container orchestration config
├── install_docker.sh      # Docker installation helper
├── wait-for-it.sh         # Service health check utility
├── mlflow/                # MLflow server configuration
│   ├── Dockerfile         # MLflow container definition
│   └── requirements.txt   # Python dependencies
└── mlruns/                # Local model storage directory
```

## Example: Working with Models

The repository already includes an example model in `mlruns/0/21f2c8adae5249609daa0d870900caff/`. 

To serve this model:

```bash
# Set the model path
export MODEL_PATH=models:/my-model/Production

# Serve the model
mlflow models serve -m $MODEL_PATH -p 1234 --no-conda
```

Test the model:

```bash
curl -X POST -H "Content-Type:application/json" \
  --data '{"dataframe_split": {"columns": ["feature1", "feature2"], "data": [[1.0, 2.0]]}}' \
  http://localhost:1234/invocations
```

## Backup and Persistence

Data is persisted in Docker volumes:
- `mlflow_db`: MySQL database
- `mlflow_artifacts`: MinIO object storage

Backup recommendations:
```bash
# Backup MySQL database
docker exec mlflow_db mysqldump -u root -p mlflow > mlflow_backup.sql

# Backup MinIO artifacts
docker run --rm -v mlflow_artifacts:/data -v $(pwd):/backup alpine tar -czvf /backup/artifacts_backup.tar.gz /data
```

## Troubleshooting

Common issues:

1. **Connection refused**: Ensure ports 5000 and 9000 are not in use
2. **Authentication failure**: Check your `.env` file and credentials
3. **Missing artifacts**: Verify MinIO is running with `docker-compose ps`
