import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the api module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.get_best_model import get_best_model

class TestGetBestModel(unittest.TestCase):
    
    @patch('api.get_best_model.mlflow')
    def test_get_best_model_success(self, mock_mlflow):
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "123456"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        # Mock runs dataframe with accuracy metrics
        mock_runs = pd.DataFrame({
            'run_id': ['run1', 'run2', 'run3'],
            'metrics.accuracy': [0.85, 0.92, 0.78]
        })
        mock_mlflow.search_runs.return_value = mock_runs
        
        # Call the function
        model_uri = get_best_model(metric_name="accuracy")
        
        # Verify the correct model URI is returned (run2 has highest accuracy)
        self.assertEqual(model_uri, "runs:/run2/model")
        
        # Verify mlflow was called correctly
        mock_mlflow.get_experiment_by_name.assert_called_once()
        mock_mlflow.search_runs.assert_called_once_with(experiment_ids=["123456"])
        
    @patch('api.get_best_model.mlflow')
    def test_get_best_model_experiment_not_found(self, mock_mlflow):
        # Mock experiment not found
        mock_mlflow.get_experiment_by_name.return_value = None
        
        # Verify the function raises an exception
        with self.assertRaises(Exception) as context:
            get_best_model(metric_name="accuracy")
            
        self.assertTrue("Experiment 'Anemia Classification Models' not found" in str(context.exception))
        
    @patch('api.get_best_model.mlflow')
    def test_get_best_model_no_runs(self, mock_mlflow):
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "123456"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        # Mock empty runs dataframe
        mock_runs = pd.DataFrame()
        mock_mlflow.search_runs.return_value = mock_runs
        
        # Verify the function raises an exception
        with self.assertRaises(Exception) as context:
            get_best_model(metric_name="accuracy")
            
        self.assertTrue("No runs found for experiment" in str(context.exception))
        
    @patch('api.get_best_model.mlflow')
    def test_get_best_model_lower_metric(self, mock_mlflow):
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "123456"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        # Mock runs dataframe with mse metrics (lower is better)
        mock_runs = pd.DataFrame({
            'run_id': ['run1', 'run2', 'run3'],
            'metrics.mse': [0.15, 0.08, 0.22]
        })
        mock_mlflow.search_runs.return_value = mock_runs
        
        # Call the function with mse metric
        model_uri = get_best_model(metric_name="mse")
        
        # Verify the correct model URI is returned (run2 has lowest mse)
        self.assertEqual(model_uri, "runs:/run2/model")
        
if __name__ == '__main__':
    unittest.main()