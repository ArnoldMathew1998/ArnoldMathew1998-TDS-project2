# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "requests",
# ]
# ///

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# AI Proxy Configuration
API_PROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}

def perform_analysis_and_visualization(data, analysis_suggestions, output_prefix):
    """
    Perform analysis and generate visualizations based on LLM suggestions.
    """
    analysis_results = {}
    os.makedirs(output_prefix, exist_ok=True)

    for column, analyses in analysis_suggestions.items():
        results = {}
        for analysis in analyses:
            try:
                print(f"Processing column: {column}, Analysis: {analysis}")

                # Summary Statistics
                if analysis == "summary statistics":
                    results["summary_statistics"] = data[column].describe().to_dict()

                # Frequency Counts
                elif analysis == "frequency counts":
                    results["frequency_counts"] = data[column].value_counts().to_dict()
                    if not data[column].dropna().empty:
                        plt.figure(figsize=(8, 6))
                        sns.countplot(x=column, data=data, order=data[column].value_counts().index[:10])
                        plt.title(f"Countplot of {column}")
                        plt.xticks(rotation=45)
                        chart_path = os.path.join(output_prefix, f"{column}_countplot.png")
                        plt.savefig(chart_path)
                        plt.close()
                        print(f"Saved countplot: {chart_path}")

                # Histogram
                elif analysis == "histogram" and pd.api.types.is_numeric_dtype(data[column]):
                    if not data[column].dropna().empty:
                        plt.figure(figsize=(8, 6))
                        sns.histplot(data[column], kde=True, bins=30)
                        plt.title(f"Histogram of {column}")
                        chart_path = os.path.join(output_prefix, f"{column}_histogram.png")
                        plt.savefig(chart_path)
                        plt.close()
                        print(f"Saved histogram: {chart_path}")

                # Correlation
                elif analysis.startswith("correlation") and pd.api.types.is_numeric_dtype(data[column]):
                    target_column = analysis.split("with")[-1].strip()
                    if target_column in data.columns and pd.api.types.is_numeric_dtype(data[target_column]):
                        correlation = data[column].corr(data[target_column])
                        results[f"correlation_with_{target_column}"] = correlation
                        print(f"Correlation between {column} and {target_column}: {correlation}")

                # Outlier Detection
                elif analysis == "outlier detection" and pd.api.types.is_numeric_dtype(data[column]):
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = data[(data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))]
                    results["outliers"] = outliers.shape[0]
                    if not data[column].dropna().empty:
                        plt.figure(figsize=(6, 4))
                        sns.boxplot(y=data[column])
                        plt.title(f"Boxplot of {column}")
                        chart_path = os.path.join(output_prefix, f"{column}_boxplot.png")
                        plt.savefig(chart_path)
                        plt.close()
                        print(f"Saved boxplot: {chart_path}")

                # Unique Values
                elif analysis == "unique values":
                    results["unique_values"] = data[column].nunique()

                # Trends Over Time
                elif analysis == "trends over time" and pd.api.types.is_datetime64_any_dtype(data[column]):
                    if not data[column].dropna().empty:
                        plt.figure(figsize=(10, 6))
                        data.groupby(column).size().plot(kind="line")
                        plt.title(f"Trends Over Time for {column}")
                        plt.xlabel(column)
                        plt.ylabel("Frequency")
                        chart_path = os.path.join(output_prefix, f"{column}_trends_over_time.png")
                        plt.savefig(chart_path)
                        plt.close()
                        print(f"Saved trends over time chart: {chart_path}")

            except Exception as e:
                print(f"Error processing column: {column}, Analysis: {analysis}, Error: {e}")
                results[f"{analysis}_error"] = str(e)

        analysis_results[column] = results
    return analysis_results

def get_basic_analysis_suggestions(column_info):
    """
    Ask the LLM for basic analysis suggestions for each column.
    """
    prompt = f"""
    You are a data analyst. Based on the following column information and sample rows, suggest the appropriate basic analysis for each column.

    Column Info:
    {column_info}

    Examples of basic analysis include:
    - For numeric columns: summary statistics, correlation, histograms, outlier detection.
    - For categorical columns: frequency counts, mode, and bar charts.
    - For text columns: word counts, unique values, and sentiment analysis.
    - For date columns: trends over time or time-based grouping.

    Return your answer as a valid JSON object where keys are column names and values are a list of suggested analysis types.
    """
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(
            f"{API_PROXY_BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload
        )
        response.raise_for_status()
        raw_content = response.json()["choices"][0]["message"]["content"]
        if raw_content.startswith("```json"):
            raw_content = raw_content.strip("```json").strip("```")
        return json.loads(raw_content)
    except Exception as e:
        print(f"Error querying AI Proxy for analysis suggestions: {e}")
        return {}

def clean_analysis_suggestions(raw_suggestions):
    """
    Clean the analysis suggestions to standardize analysis types.
    """
    cleaned_suggestions = {}
    for column, analyses in raw_suggestions.items():
        cleaned_analyses = []
        for analysis in analyses:
            # Remove text in parentheses and strip whitespace
            analysis = analysis.split('(')[0].strip()
            cleaned_analyses.append(analysis)
        cleaned_suggestions[column] = cleaned_analyses
    return cleaned_suggestions

def get_story_from_analysis(analysis_results):
    """
    Ask the LLM to write a story based on the analysis results.
    """
    prompt = f"""
    Based on the following analysis results, write a story about the dataset. Include:
    1. A brief overview of the dataset.
    2. Key findings from the analysis.
    3. Insights or implications of the findings.
    4. Recommendations based on the findings.

    Analysis Results:
    {json.dumps(analysis_results, indent=2)}
    """
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(
            f"{API_PROXY_BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error querying AI Proxy for story generation: {e}")
        return "Error generating story."

def truncate_analysis_results(analysis_results, max_columns=5, max_entries_per_column=5):
    """
    Truncate analysis_results to limit the number of columns and entries per column.
    """
    truncated_results = {}
    for i, (column, analyses) in enumerate(analysis_results.items()):
        if i >= max_columns:
            break
        truncated_analyses = {}
        for analysis, result in analyses.items():
            if isinstance(result, dict):  # For dictionary-like results
                truncated_analyses[analysis] = dict(list(result.items())[:max_entries_per_column])
            else:
                truncated_analyses[analysis] = result
        truncated_results[column] = truncated_analyses
    return truncated_results


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]

    # Attempt to read the file with proper encoding
    try:
        data = pd.read_csv(filename, encoding="utf-8")
    except UnicodeDecodeError:
        print("UTF-8 decoding failed. Attempting 'latin1' encoding...")
        data = pd.read_csv(filename, encoding="latin1")

    output_prefix = os.path.splitext(os.path.basename(filename))[0]
    os.makedirs(output_prefix, exist_ok=True)

    print("Extracting column information...")
    column_info = {
        "columns": list(data.columns),
        "types": data.dtypes.astype(str).to_dict(),
        "sample_rows": data.head(5).to_dict(orient="records"),
    }

    print("Querying the AI Proxy for basic analysis suggestions...")
    raw_suggestions = get_basic_analysis_suggestions(column_info)
    print("Raw suggestions received:", raw_suggestions)

    # Clean the analysis suggestions
    analysis_suggestions = clean_analysis_suggestions(raw_suggestions)
    print("Cleaned analysis suggestions:", analysis_suggestions)

    print("Performing analysis and generating visualizations...")
    analysis_results = perform_analysis_and_visualization(data, analysis_suggestions, output_prefix)
    print("Analysis and visualizations completed.")

    # Truncate analysis_results
    print("Truncating analysis results for story generation...")
    truncated_results = truncate_analysis_results(analysis_results, max_columns=5, max_entries_per_column=5)
    print(f"Truncated analysis results: {truncated_results}")

    print("Querying the AI Proxy for story generation...")
    story = get_story_from_analysis(truncated_results)

    output_file = os.path.join(output_prefix, "README.md")
    with open(output_file, "w") as f:
        f.write(story)

    print("Analysis complete. Check output directory for results.")


if __name__ == "__main__":
    main()
