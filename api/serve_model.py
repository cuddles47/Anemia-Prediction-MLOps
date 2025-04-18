"""
Script to serve a model using MLflow's built-in REST API capabilities.
"""
import os
import argparse
import mlflow.pyfunc
from api.get_best_model import get_best_model

def serve_model(model_uri=None, port=5001, host="0.0.0.0", workers=1, metric_name="accuracy"):
    """
    Serve a model using MLflow's built-in REST API capabilities.
    
    Args:
        model_uri (str): The URI of the model to serve
        port (int): The port to serve the model on
        host (str): The host to serve the model on
        workers (int): The number of workers to use
        metric_name (str): The metric to use for finding the best model if model_uri is not provided
    """
    if model_uri is None:
        print("Model URI not provided, finding best model based on accuracy...")
        model_uri = get_best_model(metric_name=metric_name)
    
    print(f"Serving model from: {model_uri}")
    print(f"The model will be available at: http://{host}:{port}/invocations")
    
    # We'll use MLflow CLI command directly for serving
    serve_command = f"mlflow models serve -m {model_uri} -p {port} -h {host} --workers {workers} --no-conda"
    
    print(f"Running command: {serve_command}")
    os.system(serve_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Serve a model using MLflow.')
    parser.add_argument('--model-uri', type=str, help='The URI of the model to serve')
    parser.add_argument('--port', type=int, default=5001, help='The port to serve the model on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='The host to serve the model on')
    parser.add_argument('--workers', type=int, default=1, help='The number of workers to use')
    parser.add_argument('--metric', type=str, default='accuracy', help='The metric to use for finding the best model')
    
    args = parser.parse_args()
    
    serve_model(
        model_uri=args.model_uri,
        port=args.port,
        host=args.host,
        workers=args.workers,
        metric_name=args.metric
    )