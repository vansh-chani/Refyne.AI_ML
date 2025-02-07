import os
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-pro")

def find_dataset(data_desc,model_name = ""):
    response = model.generate_content(f"I am building an {model_name} for which i want you to find me urls for {data_desc} datasets(atleast 3) from the internet. return those urls in this form:URL1, URL2, URL3, ...")
    return response.parts

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

