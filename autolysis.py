import os
import sys
import pandas as pd

def load_token():
    token = os.getenv("AIPROXY_TOKEN")
    if not token:
        raise ValueError("Environment variable 'AIPROXY_TOKEN' is not set.")
    print(f"Token loaded successfully: {token}")
    return token

def process_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded CSV: {file_path}")
        print(f"Data Preview:\n{data.head()}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error while reading the CSV file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py /path/to/input.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    token = load_token()
    process_csv(csv_path)
