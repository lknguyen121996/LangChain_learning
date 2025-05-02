import os
import requests
import pprint

from dotenv import load_dotenv

load_dotenv()

def get_linkedin_profile():
    url = f"https://gist.githubusercontent.com/emarco177/859ec7d786b45d8e3e3f688c6c9139d8/raw/5eaf8e46dc29a98612c8fe0c774123a7a2ac4575/eden-marco-scrapin.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
