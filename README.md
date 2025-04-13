# MLflow Project

This repository contains scripts and configuration files for setting up an MLflow tracking server using Docker and deploying machine learning models.

## Project Structure

- `anaconda_install.sh`: Script to install Anaconda
- `deploy_mlflow.sh`: Script to deploy MLflow server
- `docker-compose.yml`: Docker Compose configuration for services
- `install_docker.sh`: Script to install Docker
- `mlflow-docker-compose.yml`: Docker Compose configuration specific to MLflow
- `test.py`: Test script
- `mlflow/`: Directory containing MLflow server setup
  - `Dockerfile`: Docker image configuration for MLflow
  - `requirements.txt`: Python dependencies for MLflow
  - `wait-for-it.sh`: Utility script for service health checks
- `mlruns/`: Directory containing ML model runs and artifacts

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   ./install_docker.sh
   ./anaconda_install.sh
   ```
3. Copy environment variables template:
   ```
   cp env.example .env
   ```
4. Edit `.env` file with your credentials
5. Deploy MLflow server:
   ```
   ./deploy_mlflow.sh
   ```

## Environment Variables

This project uses environment variables for configuration. Copy `env.example` to `.env` and update with your actual values:

- AWS credentials for S3 storage
- MySQL database configuration

## Note

The `.env` file contains sensitive information and is excluded from version control via `.gitignore`. Always use the `env.example` template as a reference for required environment variables.

## MLflow UI

Once deployed, the MLflow UI can be accessed from your browser.