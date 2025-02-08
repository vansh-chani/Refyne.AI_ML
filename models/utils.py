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
