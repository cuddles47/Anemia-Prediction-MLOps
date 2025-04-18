import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the parent directory to sys.path to import the api module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.serve_model import serve_model

class TestServeModel(unittest.TestCase):
    
    @patch('api.serve_model.get_best_model')
    @patch('api.serve_model.os.system')
    def test_serve_model_with_uri(self, mock_system, mock_get_best_model):
        # Test serving model with a provided URI
        model_uri = "runs:/some-run-id/model"
        port = 5000
        host = "127.0.0.1"
        workers = 2
        
        serve_model(model_uri=model_uri, port=port, host=host, workers=workers)
        
        # Verify get_best_model was not called
        mock_get_best_model.assert_not_called()
        
        # Verify the correct command was executed
        expected_cmd = f"mlflow models serve -m {model_uri} -p {port} -h {host} --workers {workers} --no-conda"
        mock_system.assert_called_once_with(expected_cmd)
    
    @patch('api.serve_model.get_best_model')
    @patch('api.serve_model.os.system')
    def test_serve_model_without_uri(self, mock_system, mock_get_best_model):
        # Test serving model without a provided URI (should call get_best_model)
        mock_get_best_model.return_value = "runs:/best-model-id/model"
        port = 5001
        host = "0.0.0.0"
        workers = 4
        metric = "f1_score"
        
        serve_model(model_uri=None, port=port, host=host, workers=workers, metric_name=metric)
        
        # Verify get_best_model was called with the right metric
        mock_get_best_model.assert_called_once_with(metric_name=metric)
        
        # Verify the correct command was executed with the best model URI
        expected_cmd = f"mlflow models serve -m runs:/best-model-id/model -p {port} -h {host} --workers {workers} --no-conda"
        mock_system.assert_called_once_with(expected_cmd)

if __name__ == '__main__':
    unittest.main()