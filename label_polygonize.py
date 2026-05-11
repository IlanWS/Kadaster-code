#Here we 

import requests
import torch
import matplotlib.pyplot as plt
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
        url = url.replace("localhost", "localhost:8080")
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


def reshape(json_path, index=0):
    # Load JSON to get the image URL
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if index >= len(data):
        print(f"Error: Index {index} out of range. JSON contains {len(data)} items.")
        return None, None, None
    
    url = data[index].get("URL")
    if not url:
        print(f"Error: No URL found at index {index}")
        return None, None, None
    
    # Download image directly from URL instead of reading from local disk
    image = download_image_from_url(url)

    if image is None:
        print(f"Error: Failed to download image from URL")
        return None, None, None
    
    # Store original dimensions
    original_width, original_height = image.size
    print(f"Original image size: {original_width}x{original_height}")
    
    # Resize image to config dimensions for inference
    image_resized = image.resize((input_image_width, input_image_height), Image.NEAREST)
    
    image_array = np.array(image_resized)

    # Handle RGB/RGBA conversion
    if image_array.shape[2] == 4:
        # Remove alpha channel
        input_image = image_array[:, :, :3]
    else:
        input_image = image_array
    
    input_image = np.expand_dims(input_image, axis=0)

    input_image = np.mean(input_image, axis=3, keepdims=True)

    input_image[input_image<10] = 0
    input_image[input_image>=10] = 1
    input_image = np.array(input_image, dtype=int)
    
    predictions = input_image

    # Resize predictions back to original image dimensions
    predictions_resized = Image.fromarray((predictions[0, :, :, 0] * 255).astype(np.uint8))
    predictions_resized = predictions_resized.resize((original_width, original_height), Image.NEAREST)
    predictions_resized = np.array(predictions_resized) / 255.0
    
    #als we binaire output willen ipv heatmap, gebruik volgende lijn.
    #predictions = (predictions > 0.5).astype(np.uint8)
    return predictions_resized, (original_width, original_height)

def polygonize_label(json_path, shapefile_path, image_index=0, threshold=0.5):
    # Load JSON for georeferencing info
    with open(json_path, 'r') as f:
        data = json.load(f)
    query = data[0]['Query']
    bbox = query['BBOX'].split('%2C')
    minx, miny, maxx, maxy = map(float, bbox)
    crs = query['CRS'].replace('%3A', ':')

    img, original_dims = reshape(json_path, image_index)
    if img is None:
        print("Error: Inference failed")
        return
    
    if original_dims is None:
        print("Error: Could not get original image dimensions")
        return
    
    original_width, original_height = original_dims
    print(f"Using original dimensions for georeference: {original_width}x{original_height}")
    
    # Use the prediction directly (already resized back to original dimensions)
    arr = img

    if arr.ndim == 3 and arr.shape[2] == 4:
        mask_arr = arr[:, :, 3]
    elif arr.ndim == 3:
        mask_arr = np.mean(arr[:, :, :3], axis=2).astype(np.uint8)
    else:
        mask_arr = arr

    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr * 255).astype(np.uint8)

    binary = (mask_arr > int(threshold * 255)).astype(np.uint8)

    # Compute georeferenced transform using original image dimensions
    pixel_width = (maxx - minx) / original_width
    pixel_height = (maxy - miny) / original_height
    transform = Affine.translation(minx, maxy) * Affine.scale(pixel_width, -pixel_height)
    

    
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


name= "kadaster"
# Example usage:
json_path = "".join(["C:\\Users\\SmeerdijkIlan\\Documents\\Master_thesis_opdracht\\Data\\JSON_files\\", name, ".json"])
shapefile_path = "".join(["label_",name,".shp"])
    
# om te runnen moet de locale server draaien met mapfile met rand
polygonize_label(json_path, shapefile_path, image_index=0, threshold=0.1)