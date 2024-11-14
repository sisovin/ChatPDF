import httpx

def test_connection():
    try:
        response = httpx.get("http://localhost:11434")  # Replace with your model endpoint
        if response.status_code == 200:
            print("Connection successful")
        else:
            print(f"Connection failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Connection failed: {e}")

test_connection()