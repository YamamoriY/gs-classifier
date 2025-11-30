import requests
import json
from pprint import pprint
from dotenv import load_dotenv
import os
from pathlib import Path
load_dotenv()

API_KEY = os.getenv("PLANTNET_API_KEY")
PROJECT = "all" # try "weurope" or "canada"
api_endpoint = f"https://my-api.plantnet.org/v2/identify/{PROJECT}?api-key={API_KEY}"

# 現在のファイルのディレクトリを基準にパスを解決
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
image_path_1 = project_root / "tmp" / "images" / "render_above_3" / "1.png"
image_data_1 = open(image_path_1, 'rb')

image_path_2 = project_root / "tmp" / "images" / "render_above_3" / "2.png"
image_data_2 = open(image_path_2, 'rb')


data = {
    'organs': ['habit', 'flower']
}

files = [
    ('images', (str(image_path_1), image_data_1)),
    ('images', (str(image_path_2), image_data_2))
]

req = requests.Request('POST', url=api_endpoint, files=files, data=data)
prepared = req.prepare()

s = requests.Session()
response = s.send(prepared)
json_result = json.loads(response.text)

pprint(response.status_code)
pprint(json_result)