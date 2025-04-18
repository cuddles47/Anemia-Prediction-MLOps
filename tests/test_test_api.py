import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the api module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.test_api import test_model_api

class TestTestAPI(unittest.TestCase):
    
    @patch('api.test_api.requests.post')
    def test_api_client_success(self, mock_post):
        # Create a mock response with successful status code
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [3, 1, 2]  # Some example predictions
        mock_post.return_value = mock_response
        
        # Call the function with test parameters
        test_model_api(host="localhost", port=5001, sample_count=3)
        
        # Verify requests.post was called with the right URL
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['headers'], {"Content-Type": "application/json"})
        self.assertEqual(args[0], "http://localhost:5001/invocations")
        
    @patch('api.test_api.requests.post')
    def test_api_client_error(self, mock_post):
        # Create a mock response with error status code
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response
        
        # Call the function with test parameters
        test_model_api(host="localhost", port=5001, sample_count=1)
        
        # Verify requests.post was called with the right URL
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[0][0], "http://localhost:5001/invocations")
        
    @patch('api.test_api.requests.post')
    def test_sample_data_generation(self, mock_post):
        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [3]
        mock_post.return_value = mock_response
        
        # Call the function to generate a single sample
        test_model_api(sample_count=1)
        
        # Verify that the request contains the expected data structure
        _, kwargs = mock_post.call_args
        payload = kwargs['data']
        
        # Check that the payload is JSON with the expected format
        self.assertIn('"columns":', payload)
        self.assertIn('"data":', payload)
        self.assertIn('"index":', payload)

if __name__ == '__main__':
    unittest.main()