# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai",
#   "tenacity",
#   "uvicorn",
# ]
# ///
import matplotlib
matplotlib.use('Agg')


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging

# Configuration
CONFIG = {
    "OUTPUT_DIR": "autolysis",
    "MAX_ROWS": 1000
}

# Initialize FastAPI
app = FastAPI()

# Logging Setup
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def detect_encoding(file_path):
    """Detect file encoding."""
    import chardet
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]

def save_visualization(plt, file_path):
    """Save a matplotlib visualization."""
    try:
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        logger.info(f"Visualization saved: {file_path}")
    except Exception as e:
        logger.error(f"Error saving visualization: {e}")

def plot_correlation_matrix(df, output_dir):
    """Generate and save a correlation heatmap."""
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        logger.warning("No numeric columns found for correlation analysis.")
        return
    correlation = numeric_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    save_visualization(plt, os.path.join(output_dir, "correlation_heatmap.png"))

def plot_outliers(df, output_dir):
    """Generate and save an outlier detection plot."""
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        logger.warning("No numeric data for outlier analysis.")
        return
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=numeric_df)
    plt.title("Outlier Detection")
    plt.xticks(rotation=90)
    save_visualization(plt, os.path.join(output_dir, "outliers.png"))

def call_openai_api(task, context):
    """Mock function for calling OpenAI API."""
    return f"Insights for task: {task}\nContext: {context[:200]}..."

def analyze_data(file_path, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Detect file encoding and load the dataset
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding, nrows=CONFIG["MAX_ROWS"])
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise RuntimeError("Failed to read dataset")

    if df.empty:
        logger.error("Dataset is empty. No analysis performed.")
        return {"message": "Dataset is empty. No analysis performed."}

    logger.info("Dataset loaded successfully.")
    summary = df.describe(include='all')
    missing_values = df.isna().sum()

    # Log missing values if present
    if missing_values.sum() > 0:
        logger.info(f"Missing values detected: {missing_values}")

    # Generate insights using the AI Proxy
    context = f"Summary statistics:\n{summary}\nMissing values:\n{missing_values}"
    insights = call_openai_api("Analyze the dataset and provide insights.", context)

    logger.info(f"Generated insights: {insights}")

    # Perform visualizations
    try:
        plot_correlation_matrix(df, output_dir)
        plot_outliers(df, output_dir)
    except Exception as e:
        logger.error(f"Error during visualization generation: {e}")

    # Save README file in the dataset-named folder
    try:
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write("# Data Analysis Report\n\n")
            f.write("## Summary Statistics\n")
            f.write(f"{summary}\n\n")
            f.write("## Missing Values\n")
            f.write(f"{missing_values}\n\n")
            f.write("## Insights\n")
            f.write(f"{insights}\n")
    except Exception as e:
        logger.error(f"Error writing README.md: {e}")

    logger.info(f"Analysis complete! Results saved in {output_dir}.")

@app.post("/run")
def run_analysis(file_path: str):
    try:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(CONFIG["OUTPUT_DIR"], dataset_name)
        response = analyze_data(file_path, output_dir)
        if response:
            return response
        return {"message": "Analysis complete!"}
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/run_all")
def run_all_datasets():
    """Run analysis for multiple predefined datasets."""
    datasets = {
        "goodreads.csv": "goodreads",
        "happiness.csv": "happiness",
        "media.csv": "media"
    }
    results = []
    for file_path, dataset_name in datasets.items():
        if os.path.exists(file_path):
            logger.info(f"Processing dataset: {file_path}")
            try:
                output_dir = os.path.join(CONFIG["OUTPUT_DIR"], dataset_name)
                analyze_data(file_path=file_path, output_dir=output_dir)
                results.append(f"Analysis complete for {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append(f"Failed to analyze {file_path}: {e}")
        else:
            logger.warning(f"Dataset {file_path} not found. Skipping.")
            results.append(f"Dataset {file_path} not found. Skipping.")
    return {"results": results}
