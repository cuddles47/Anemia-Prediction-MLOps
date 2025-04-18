"""
Script to get the best model from MLflow based on a specified metric.
"""
import os
import mlflow
from mlflow.tracking import MlflowClient

def get_best_model(metric_name="accuracy", experiment_name="Anemia Classification Models"):
    """
    Get the best model from MLflow based on the specified metric.
    
    Args:
        metric_name (str): The metric to use for comparison (default: "accuracy")
        experiment_name (str): Name of the experiment to search in
        
    Returns:
        The URI of the best model
    """
    # Set MLflow tracking URI
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Get experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise Exception(f"Experiment '{experiment_name}' not found")
    
    # Get all runs in the experiment
    client = MlflowClient()
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        raise Exception(f"No runs found for experiment '{experiment_name}'")
    
    # Find the run with the best metric value
    if metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
        # For these metrics, higher is better
        best_run = runs.loc[runs[f"metrics.{metric_name}"].idxmax()]
    else:
        # For metrics like MSE, RMSE, etc., lower is better
        best_run = runs.loc[runs[f"metrics.{metric_name}"].idxmin()]
    
    # Get the run ID of the best run
    best_run_id = best_run.run_id
    
    # Get the model URI
    model_uri = f"runs:/{best_run_id}/model"
    
    print(f"Best model found with {metric_name}: {best_run[f'metrics.{metric_name}']}")
    print(f"Model URI: {model_uri}")
    
    return model_uri

if __name__ == "__main__":
    # Get the best model based on accuracy
    model_uri = get_best_model(metric_name="accuracy")
    print(f"Best model URI: {model_uri}")