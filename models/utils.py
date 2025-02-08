import os
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess

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
        print(f"❌ Error fetching Kaggle datasets: {result.stderr}")
        return []

    lines = result.stdout.strip().split("\n")[1:]  # Skip header
    datasets = [line.split()[0] for line in lines if line]  # Extract dataset names

    return datasets

# Function to download datasets
def download_datasets(dataset_list):
    download_path = "./kaggle_downloads"
    
    for dataset in dataset_list:
        print(f"📥 Downloading: {dataset} ...")
        result = subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset, '--unzip', '-p', download_path], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Error downloading {dataset}: {result.stderr}")
        else:
            print(f"✅ Downloaded: {dataset}")

def find_dataset(data_desc, model_name=""):
    search_query = data_desc
    
    try:
        datasets = get_kaggle_datasets(search_query)

        if not datasets:
            print("❌ No datasets found!")
            return

        print(f"🔍 Found {len(datasets)} datasets")
        
        # Take only the first 2 datasets
        selected_datasets = datasets[:3]
        
        print(f"⏳ Downloading first 2 datasets: {selected_datasets}")
        download_datasets(selected_datasets)
        print("✅ First 2 datasets downloaded successfully!")

    except Exception as e:
        print(f"❌ Error in find_dataset: {str(e)}")



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

print(find_dataset("Housing prices","linear Regression"))