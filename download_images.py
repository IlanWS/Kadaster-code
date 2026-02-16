import json
import os
import requests

json_path = "zutphen-met-labels.json"
output_folder = "zutphen-zonder-gebouwen-map"
timeout_seconds = 5

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
        save_path = os.path.join(output_folder, filename)
        response = requests.get(url, timeout=timeout_seconds)
        if response.status_code == 200:
            with open(save_path, 'wb') as img_file:
                img_file.write(response.content)
            print(f"Succes: {filename} opgeslagen.")
        else:
            print(f"Fout bij {url}: Status {response.status_code}")
    except Exception as e:
        print(f"Fout bij downloaden van {url}: {e}")
