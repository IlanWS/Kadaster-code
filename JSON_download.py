#only needs to be ran if new data must be obtained from QGIS through the WSL server
import json
import os
import requests
from config import *

json_path = json_path
input_folder = input_folder
output_folder = output_folder
timeout_seconds = 5

def json_download(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Start met downloaden van {len(data)} afbeeldingen...")

    for index, item in enumerate(data):
        url = item.get("URL")
        if not url:
            print(f"Skipping index {index}: Geen URL gevonden.")
            continue
        try:
            filename = f"image_{index}.jpg"
            save_path = os.path.join(folder, filename)
            response = requests.get(url, timeout=timeout_seconds)
            if response.status_code == 200:
                with open(save_path, 'wb') as img_file:
                    img_file.write(response.content)
                print(f"Succes: {filename} opgeslagen.")
            else:
                print(f"Fout bij {url}: Status {response.status_code}")
        except Exception as e:
            print(f"Fout bij downloaden van {url}: {e}")


#this must first be ran, while the WSL is configured for the roadnetwork (with input_data_dir)
#then it must be ran again with the WSL configured for the labels (with output_data)
json_download(input_data_dir)
