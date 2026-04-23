#Here we create a workflow to inference a shapefile from only the roadnetwork as input data
#We start by running the input roadnetwork through the model, then we take the output and polygonize it add the geolocation data
#we use the amsterdam image and the stacked hourglass model with 100 epochs
import requests
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from PIL import Image
from affine import Affine
from Load_model import predict_model
import rasterio.features
import fiona
from shapely.geometry import shape, mapping
from shapely.validation import make_valid
import json
import io

from Model import *
from config import *

def download_image_from_url(url, timeout_seconds=5):
    try:
        response = requests.get(url, timeout=timeout_seconds)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            print(f"Successfully downloaded image from URL")
            return image
        else:
            print(f"Error downloading from URL: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading from URL: {e}")
        return None


def inference(json_path, index=0):
    # Load JSON to get the image URL
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if index >= len(data):
        print(f"Error: Index {index} out of range. JSON contains {len(data)} items.")
        return None, None
    
    url = data[index].get("URL")
    if not url:
        print(f"Error: No URL found at index {index}")
        return None, None
    
    # Download image directly from URL instead of reading from local disk
    image = download_image_from_url(url)
    if image is None:
        print(f"Error: Failed to download image from URL")
        return None, None
    
    image_array = np.array(image)
    input_image = np.delete(np.delete(np.delete(image_array, np.s_[input_image_height::], 0),np.s_[input_image_width::], 1),np.s_[3::], 2)
    input_image = np.expand_dims(input_image, axis=0)

    input_image = np.mean(input_image, axis = 3, keepdims = True)
    input_image[input_image<128] = 0
    input_image[input_image>=128] = 1
    input_image = np.array(input_image, dtype = int)

    # Convert test data to tensor
    test_input_pt = torch.from_numpy(input_image).permute(0, 3, 1, 2).float()
    test_loader = DataLoader(test_input_pt, batch_size=1, shuffle=False)

    # Load model
    model = StackedHourglassRoadLabeler(1)
    model.load_state_dict(torch.load("C:\\Users\\SmeerdijkIlan\\Documents\\Master_thesis_opdracht\\Models\\StackedHourglass_combined_100.pth", map_location='cpu'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Clear GPU cache to free fragmented memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.time()
    # Process predictions in batches
    predictions_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch_predictions = model(batch)
            predictions_list.append(batch_predictions.cpu())

    # Concatenate all predictions and convert to numpy
    predictions = torch.cat(predictions_list, dim=0).numpy().transpose(0, 2, 3, 1)

    #als we binaire output willen ipv heatmap, gebruik volgende lijn.
    #predictions = (predictions > 0.5).astype(np.uint8)
    return predictions[0], start_time



def polygonize_results(json_path, shapefile_path, image_index=0, threshold=0.5):
    # Load JSON for georeferencing info
    with open(json_path, 'r') as f:
        data = json.load(f)
    query = data[0]['Query']
    bbox = query['BBOX'].split('%2C')
    minx, miny, maxx, maxy = map(float, bbox)
    crs = query['CRS'].replace('%3A', ':')
    width = int(query['WIDTH'])
    height = int(query['HEIGHT'])

    img, start_time = inference(json_path, image_index)
    if img is None:
        print("Error: Inference failed")
        return
    
    arr = img[:,:,0]

    if arr.ndim == 3 and arr.shape[2] == 4:
        mask_arr = arr[:, :, 3]
    elif arr.ndim == 3:
        mask_arr = np.mean(arr[:, :, :3], axis=2).astype(np.uint8)
    else:
        mask_arr = arr

    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr * 255).astype(np.uint8)

    binary = (mask_arr > int(threshold * 255)).astype(np.uint8)

    # Compute georeferenced transform
    pixel_width = (maxx - minx) / width
    pixel_height = (maxy - miny) / height
    transform = Affine.translation(minx, maxy) * Affine.scale(pixel_width, -pixel_height)
    
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    
    schema = {
        "geometry": "Polygon",
        "properties": {"value": "int:10"},
    }

    with fiona.open(
        shapefile_path,
        mode="w",
        driver="ESRI Shapefile",
        schema=schema,
        crs=crs,
    ) as shp:
        for geom, value in rasterio.features.shapes(binary, mask=binary, transform=transform):
            if value == 1:
                # Validate and fix invalid geometries
                geom_shape = shape(geom)
                if not geom_shape.is_valid:
                    geom_shape = make_valid(geom_shape)
                
                if geom_shape.is_empty:
                    continue
                
                shp.write({"geometry": mapping(geom_shape), "properties": {"value": int(value)}})


name= "keizer"
# Example usage:
json_path = "".join(["C:\\Users\\SmeerdijkIlan\\Documents\\Master_thesis_opdracht\\Data\\JSON_files\\", name, ".json"])
shapefile_path = "".join(["prediction_",name,".shp"])
    
# om te runnen moet de locale server draaien met mapfile met rand
polygonize_results(json_path, shapefile_path, image_index=0, threshold=0.5)