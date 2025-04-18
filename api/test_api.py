~"""
Script to test the model API by sending sample requests.
"""
import json
import requests
import pandas as pd
import numpy as np

def test_model_api(host="localhost", port=5001, sample_count=1):
    """
    Test the model API by sending sample requests.
    
    Args:
        host (str): The host where the model is being served
        port (int): The port the model is being served on
        sample_count (int): Number of random samples to generate
    """
    # Define the URL for the model API
    url = f"http://{host}:{port}/invocations"
    
    # Create a sample input (adjust these values based on your model's expected input)
    sample_data = {
        "age": np.random.uniform(15, 45, sample_count).tolist(),
        "births_5_years": np.random.randint(1, 4, sample_count).tolist(),
        "respondent_1st_birth": np.random.randint(13, 30, sample_count).tolist(),
        "hemoglobin_altitude_smoking": np.random.uniform(90, 150, sample_count).tolist(),
        "mosquito_bed_sleeping": np.random.randint(0, 2, sample_count).tolist(),
        "smoking": np.random.randint(0, 2, sample_count).tolist(),
        "status": np.random.randint(1, 5, sample_count).tolist(),
        "residing_husband_partner": np.random.randint(0, 2, sample_count).tolist(),
        "child_put_breast": np.random.uniform(0, 100, sample_count).tolist(),
        "fever_two_weeks": np.random.randint(0, 2, sample_count).tolist(),
        "hemoglobin_altitude": np.random.uniform(70, 130, sample_count).tolist(),
        "anemia_level_1": np.random.randint(0, 5, sample_count).tolist(),
        "iron_pills": np.random.randint(0, 2, sample_count).tolist(),
        "residence_Rural": np.random.randint(0, 2, sample_count).astype(bool).tolist(),
        "residence_Urban": np.random.randint(0, 2, sample_count).astype(bool).tolist(),
        "highest_educational_Higher": np.random.randint(0, 2, sample_count).astype(bool).tolist(),
        "highest_educational_No education": np.random.randint(0, 2, sample_count).astype(bool).tolist(),
        "highest_educational_Primary": np.random.randint(0, 2, sample_count).astype(bool).tolist(),
        "highest_educational_Secondary": np.random.randint(0, 2, sample_count).astype(bool).tolist(),
        "wealth_index_Middle": np.random.randint(0, 2, sample_count).astype(bool).tolist(),
        "wealth_index_Poorer": np.random.randint(0, 2, sample_count).astype(bool).tolist(),
        "wealth_index_Poorest": np.random.randint(0, 2, sample_count).astype(bool).tolist(),
        "wealth_index_Richer": np.random.randint(0, 2, sample_count).astype(bool).tolist(),
        "wealth_index_Richest": np.random.randint(0, 2, sample_count).astype(bool).tolist(),
    }
    
    # Convert to DataFrame and then to the format expected by MLflow serving
    df = pd.DataFrame(sample_data)
    payload = df.to_json(orient="split")
    
    # Set the headers for the request
    headers = {"Content-Type": "application/json"}
    
    print(f"Sending request to {url}")
    print(f"Sample data (first row):")
    for key, value in {k: v[0] if isinstance(v, list) and len(v) > 0 else v for k, v in sample_data.items()}.items():
        print(f"  {key}: {value}")
    
    # Send the request to the model API
    try:
        response = requests.post(url, data=payload, headers=headers)
        
        # Check the response
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction successful!")
            print(f"Predicted values: {result}")
            
            # Map the numeric predictions back to categories
            anemia_map = {0: 'Dont know', 1: 'Moderate', 2: 'Mild', 3: 'Not anemic', 4: 'Severe'}
            if isinstance(result, list):
                mapped_results = [anemia_map.get(int(r), f"Unknown: {r}") for r in result]
                print("\nMapped predictions:")
                for i, prediction in enumerate(mapped_results):
                    print(f"Sample {i+1}: {prediction}")
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error sending request: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the model API.')
    parser.add_argument('--host', type=str, default='localhost', help='The host where the model is being served')
    parser.add_argument('--port', type=int, default=5001, help='The port the model is being served on')
    parser.add_argument('--samples', type=int, default=3, help='Number of random samples to generate')
    
    args = parser.parse_args()
    
    test_model_api(host=args.host, port=args.port, sample_count=args.samples)