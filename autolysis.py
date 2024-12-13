# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chardet",
#     "pandas",
#     "seaborn",
#     "matplotlib",
#     "requests",
# ]
# ///

import os
import sys
import subprocess
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import requests
import json

# Install missing dependencies
def ensure_dependencies():
    required_packages = ["chardet", "seaborn", "matplotlib", "pandas", "requests"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"'{package}' module not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure all dependencies are installed
ensure_dependencies()
import chardet  # Safe to import after installation check

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {AIPROXY_TOKEN}",
    "Content-Type": "application/json"
}

def load_data(file_path):
    """Load CSV data with encoding detection."""
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected file encoding: {encoding}")
    return pd.read_csv(file_path, encoding=encoding)

def analyze_data(df):
    """Perform basic data analysis."""
    if df.empty:
        print("Error: Dataset is empty.")
        sys.exit(1)
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict()  # Compute correlation only on numeric columns
    }
    print("Data analysis complete.")
    return analysis

def create_visualizations(df, output_dir):
    """Generate and save visualizations."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns
    os.makedirs(output_dir, exist_ok=True)
    created_charts = []

    for column in numeric_columns:
        try:
            plt.figure()
            sns.histplot(df[column].dropna(), kde=True)
            plt.title(f'Distribution of {column}')
            file_name = os.path.join(output_dir, f'{column}_distribution.png')
            plt.savefig(file_name)
            created_charts.append(file_name)
            plt.close()
        except Exception as e:
            print(f"Error creating visualization for {column}: {e}")

    return created_charts

def generate_narrative(analysis):
    """Generate narrative using LLM."""
    prompt = f"Provide a detailed analysis based on the following data summary: {json.dumps(analysis)}"
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.RequestException as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return "Narrative generation failed due to an error."

def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = os.path.splitext(file_path)[0]

    print("Starting autolysis process...")
    df = load_data(file_path)
    print("Dataset loaded successfully.")

    print("Analyzing data...")
    analysis = analyze_data(df)

    print("Generating visualizations...")
    created_charts = create_visualizations(df, output_dir)
    print(f"Charts created: {created_charts}")

    print("Generating narrative...")
    narrative = generate_narrative(analysis)

    if narrative != "Narrative generation failed due to an error.":
        output_file = os.path.join(output_dir, 'README.md')
        with open(output_file, 'w') as f:
            f.write(narrative)
        print(f"Narrative successfully written to {output_file}.")
    else:
        print("Narrative generation failed. Skipping README creation.")

    print("Autolysis process completed.")

if __name__ == "__main__":
    main()
