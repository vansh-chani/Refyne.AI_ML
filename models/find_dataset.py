import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-pro")

def find_dataset(data_desc,model_name = ""):
    response = model.generate_content(f"I am building an {model_name} for which i want you to find me urls for {data_desc} datasets(atleast 3) from the internet. return those urls in this form:URL1, URL2, URL3, ...")
    return response.parts

