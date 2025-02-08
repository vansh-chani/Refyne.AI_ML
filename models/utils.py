import os
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess
import json

load_dotenv()

api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-pro")


api = KaggleApi()
api.authenticate()

import subprocess

def get_kaggle_datasets(query):
    result = subprocess.run(['kaggle', 'datasets', 'list', '-s', query], capture_output=True, text=True)

    if result.returncode != 0:
        return []

    lines = result.stdout.split("\n")
    datasets = []
    for line in lines[2:]:  
        parts = line.split()
        if len(parts) > 0:
            dataset_slug = parts[0] 
            datasets.append(dataset_slug)


    return datasets

def download_datasets(dataset_list):
    download_path = "./kaggle_downloads"
    
    for dataset in dataset_list:
        subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset, '--unzip', '-p', download_path], capture_output=True, text=True)


def find_dataset(data_desc, model_name=""):
    search_query = data_desc
    datasets = get_kaggle_datasets(search_query)
    selected_datasets = datasets[1:3]
    
    download_datasets(selected_datasets)


def combine_dataset(*dfs):
    features = {}
    name = "df"
    i = 0
    for df in dfs:
        name = name + str(i)
        features[name] = list(df.columns)
        i+=1
        name = 'df'

    response = model.generate_content(f"i wish to combine multiple datasets with same feature which have different coloumn names. these will be given to you in this form {features}. You need to group similar coloumns together and return to me in form a library like:{{\"coloumn_name\" : [list of similar coloumns in other datasets]}}i do not require any code just return to me what you think the output should be")
    return response.parts

########################


import json

def parse_gemini_response(response):
    """
    Parses the Gemini AI response into a dictionary.

    Parameters:
        response (RepeatedComposite): Response object from Gemini.

    Returns:
        dict: Parsed column mapping.
    """
    try:
        response_text = response[0].text  # Extract response

        print("Gemini Response:\n", response_text)  # Debugging line

        column_mapping = json.loads(response_text)  # Convert to dictionary
        return column_mapping
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        print("Response Received:", response_text)  # Print response for debugging
        return {}

    except Exception as e:
        print("Error parsing Gemini response:", e)
        return {}


def merge_datasets(dfs):
    """
    Merges multiple datasets by identifying similar columns using Gemini AI.

    Parameters:
        dfs (list of pd.DataFrame): List of datasets to merge.

    Returns:
        pd.DataFrame: Merged dataset.
    """
    # Step 1: Get column mapping from Gemini
    raw_response = combine_dataset(*dfs)
    column_mapping = parse_gemini_response(raw_response)

    # Step 2: Rename columns in each dataset
    for col, similar_cols in column_mapping.items():
        for df in dfs:
            for similar_col in similar_cols:
                if similar_col in df.columns:
                    df.rename(columns={similar_col: col}, inplace=True)

    # Step 3: Find common columns across all datasets
    common_columns = set(dfs[0].columns)
    for df in dfs[1:]:
        common_columns.intersection_update(set(df.columns))

    # Step 4: Merge datasets on common columns
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)[list(common_columns)]

    return merged_df


# Step 1: Find and download datasets
find_dataset("housing price")

# Step 2: Load datasets into pandas
download_path = "./kaggle_downloads"
file_paths = [os.path.join(download_path, f) for f in os.listdir(download_path) if f.endswith(".csv")]

dfs = [pd.read_csv(file) for file in file_paths]

# Step 3: Merge datasets using the function
final_df = merge_datasets(dfs)

# Step 4: Save the merged dataset
final_df.to_csv("final.csv", index=False)
print("Merged dataset saved as final.csv")
