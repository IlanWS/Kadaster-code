import json
from pathlib import Path

import requests


def download_images(
    json_path: str, output_folder: str, timeout_seconds: int = 5
) -> None:
    """Download images from WMS service based on JSON file."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Start met downloaden van {len(data)} afbeeldingen...")

    for index, item in enumerate(data):
        url = item.get("URL")
        if not url:
            print(f"Skipping index {index}: Geen URL gevonden.")
            continue
        try:
            filename = f"image_{index}.jpg"
            save_path = output_path / filename
            response = requests.get(url, timeout=timeout_seconds)
            if response.status_code == 200:
                with open(save_path, "wb") as img_file:
                    img_file.write(response.content)
                print(f"Succes: {filename} opgeslagen.")
            else:
                print(f"Fout bij {url}: Status {response.status_code}")
        except Exception as e:
            print(f"Fout bij downloaden van {url}: {e}")
