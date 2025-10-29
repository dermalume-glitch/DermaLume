import requests

# Test if the server is running
try:
    response = requests.get("http://127.0.0.1:5000/")
    print(f"Home page status: {response.status_code}")
except Exception as e:
    print(f"Cannot connect to server: {e}")

# Test the predict endpoint with a simple request
try:
    response = requests.post("http://127.0.0.1:5000/predict")
    print(f"Predict endpoint status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Cannot connect to predict endpoint: {e}")