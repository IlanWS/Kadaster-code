#Here we create a workflow to inference a shapefile from only the roadnetwork as input data
#We start by running the input roadnetwork through the model, then we take the output and polygonize it add the geolocation data
#we use the amsterdam image and the stacked hourglass model with 50 epochs
from Model import *
from config import *

import requests
import torch
import time
import numpy as np
from PIL import Image
from affine import Affine
import rasterio.features
import fiona
from shapely.geometry import shape, mapping
from shapely.validation import make_valid
import json
import io
from shapely.ops import unary_union

def download_image_from_url(url, timeout_seconds=5):
    try:
        response = requests.get(url, timeout=timeout_seconds)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            return image
        else:
            print(f"Error downloading from URL: Status {response.status_code}")
            print(f"URL: {url}")
            return None
    except Exception as e:
        print(f"Error downloading from URL: {e}")
        print(f"URL: {url}")
        return None


def inference(json_path, index=0):
    #load JSON to get the image URL
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if index >= len(data):
        print(f"Error: Index {index} out of range. JSON contains {len(data)} items.")
        return None, None, None
    
    url = data[index].get("URL")
    if not url:
        print(f"Error: No URL found at index {index}")
        return None, None, None
    
    #road network on port 80 (standard), labels on 8080
    if url.startswith("http://localhost:8080/"):
        url = url.replace("localhost:8080/", "localhost/")

    image = download_image_from_url(url)
    if image is None:
        print(f"Error: Failed to download image from URL")
        return None, None, None
    
    #load in model
    if model_name.lower().startswith("deeplab"):
        #result of hyperparameter tuning
        model = DeepLabRoadLabeler(1, use_aux=False)
    elif model_name.lower().startswith("stackedhourglass"):
        model = StackedHourglassRoadLabeler(1)
    elif model_name.lower().startswith("unet"):
        model = UNetRoadLabeler(1, 1)
    else:
        print("not a valid model name, check config.py")
        exit(1)

    model.load_state_dict(torch.load("".join([os.getcwd(), "/Models/", model_name, "_dice_50", ".pth"]), map_location='cpu'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    start_time = time.time()

    original_width, original_height = image.size
    image_resized = image.resize((input_image_width, input_image_height), Image.LANCZOS)
    
    image_array = np.array(image_resized)
    
    # RGB/RGBA conversion
    if image_array.shape[2] == 4:
        # remove alpha channel
        input_image = image_array[:, :, :3]
    else:
        input_image = image_array
    
    input_image = np.expand_dims(input_image, axis=0)

    input_image = np.mean(input_image, axis=3, keepdims=True)
    input_image[input_image<128] = 0
    input_image[input_image>=128] = 1
    input_image = np.array(input_image, dtype=int)

    test_input_pt = torch.from_numpy(input_image).permute(0, 3, 1, 2).float()
    test_loader = DataLoader(test_input_pt, batch_size=1, shuffle=False)

    #inference
    model.to(device)
    model.eval()

    #Clear GPU cache to free fragmented memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time_inference = time.time()
    
    # Look into if it would be faster (or even possible) to do without batching, only 1 image
    predictions_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch_predictions = model(batch)
            predictions_list.append(batch_predictions.cpu())

    end_time_inference = time.time()
    predictions = torch.cat(predictions_list, dim=0).numpy().transpose(0, 2, 3, 1)

    #resize predictions back to original image dimensions
    predictions_resized = Image.fromarray((predictions[0, :, :, 0] * 255).astype(np.uint8))
    predictions_resized = predictions_resized.resize((original_width, original_height), Image.LANCZOS)
    predictions_resized = np.array(predictions_resized) / 255.0

    #als we binaire output willen ipv heatmap, gebruik volgende lijn.
    #predictions = (predictions > 0.5).astype(np.uint8)
    return predictions_resized, start_time, (original_width, original_height), end_time_inference - start_time_inference


def get_labels(json_path, index=0):
    #Same documantation as function above
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if index >= len(data):
        print(f"Error: Index {index} out of range. JSON contains {len(data)} items.")
        return None, None
    
    url = data[index].get("URL")
    if not url:
        print(f"Error: No URL found at index {index}")
        return None, None
        
    if url.startswith("http://localhost/"):
        url = url.replace("localhost/", "localhost:8080/")

    image = download_image_from_url(url)

    if image is None:
        print(f"Error: Failed to download image from URL")
        return None, None
    
    original_width, original_height = image.size
    image_resized = image.resize((input_image_width, input_image_height), Image.NEAREST)
    
    image_array = np.array(image_resized)

    if image_array.shape[2] == 4:
        input_image = image_array[:, :, :3]
    else:
        input_image = image_array
    
    input_image = np.expand_dims(input_image, axis=0)
    input_image = np.mean(input_image, axis=3, keepdims=True)
    input_image[input_image<10] = 0
    input_image[input_image>=10] = 1
    input_image = np.array(input_image, dtype=int)
    
    predictions = input_image
    predictions_resized = Image.fromarray((predictions[0, :, :, 0] * 255).astype(np.uint8))
    predictions_resized = predictions_resized.resize((original_width, original_height), Image.NEAREST)
    predictions_resized = np.array(predictions_resized) / 255.0
    
    return predictions_resized, (original_width, original_height)


def polygonize_with_overlap_scores(json_path, label_shapefile_path, image_index=0, alpha=0.5, beta=0.5, prediction_shapefile_path=None):
    with open(json_path, 'r') as f:
        data = json.load(f)
    query = data[0]['Query']
    bbox = query['BBOX'].split('%2C')
    minx, miny, maxx, maxy = map(float, bbox)
    crs = query['CRS'].replace('%3A', ':')

    # Get predictions
    predictions_img, start_time, original_dims, inference_time = inference(json_path, image_index)
    if predictions_img is None:
        print("Error: Inference failed")
        return
    
    # Get labels 
    labels_img, label_dims = get_labels(json_path, image_index)
    if labels_img is None:
        print("Error: Label loading failed")
        return
    
    original_width, original_height = original_dims
    label_width, label_height = label_dims
    print(f"Using prediction dimensions for georeference: {original_width}x{original_height}")
    print(f"Using label dimensions: {label_width}x{label_height}")
    
    # Create binary masks, could prob do this in one function called 2 times but prob barely more efficient 
    pred_arr = predictions_img
    if pred_arr.ndim == 3:
        pred_mask = np.mean(pred_arr[:, :, :3], axis=2).astype(np.uint8)
    else:
        pred_mask = pred_arr
    if pred_mask.dtype != np.uint8:
        pred_mask = (pred_mask * 255).astype(np.uint8)
    #THRESHOLD FUNCTION: only include pixels above alpha
    binary_predictions = (pred_mask > int(alpha * 255)).astype(np.uint8)
    
    label_arr = labels_img
    if label_arr.ndim == 3:
        label_mask = np.mean(label_arr[:, :, :3], axis=2).astype(np.uint8)
    else:
        label_mask = label_arr
    if label_mask.dtype != np.uint8:
        label_mask = (label_mask * 255).astype(np.uint8)
    binary_labels = (label_mask > int(0.5 * 255)).astype(np.uint8)
    
    # Predictions transform
    pixel_width_pred = (maxx - minx) / original_width
    pixel_height_pred = (maxy - miny) / original_height
    transform_pred = Affine.translation(minx, maxy) * Affine.scale(pixel_width_pred, -pixel_height_pred)
    
    # Labels transform 
    pixel_width_label = (maxx - minx) / label_width
    pixel_height_label = (maxy - miny) / label_height
    transform_label = Affine.translation(minx, maxy) * Affine.scale(pixel_width_label, -pixel_height_label)
    
    #Extract prediction geometries and merge into single geometry
    prediction_polygons = []
    for geom, value in rasterio.features.shapes(binary_predictions, mask=binary_predictions, transform=transform_pred):
        if value == 1:
            geom_shape = shape(geom)
            if not geom_shape.is_valid:
                geom_shape = make_valid(geom_shape)
            if not geom_shape.is_empty:
                prediction_polygons.append(geom_shape)
    
    # Merge
    if prediction_polygons:
        all_predictions = unary_union(prediction_polygons)
    else:
        all_predictions = None
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.3f} seconds, Inference time: {inference_time:.3f} seconds")
    
    schema = {
        "geometry": "Polygon",
        "properties": {"score": "float:10.6"},
    }

    with fiona.open(
        label_shapefile_path,
        mode="w",
        driver="ESRI Shapefile",
        schema=schema,
        crs=crs,
    ) as shp:
        for geom, value in rasterio.features.shapes(binary_labels, mask=binary_labels, transform=transform_label):
            if value == 1:
                geom_shape = shape(geom)
                if not geom_shape.is_valid:
                    geom_shape = make_valid(geom_shape)
                if geom_shape.is_empty:
                    continue
                
                # Calculate overlap score
                if all_predictions is not None:
                    intersection = geom_shape.intersection(all_predictions)
                    intersection_area = intersection.area
                    label_area = geom_shape.area
                    score = intersection_area / label_area if label_area > 0 else 0.0
                else:
                    score = 0.0
                
                # OMMISSION FUNCTION: Only write if score is bigger than beta
                if score >= beta:
                    shp.write({"geometry": mapping(geom_shape), "properties": {"score": float(score)}})
    
    if prediction_shapefile_path is not None:
        pred_schema = {
            "geometry": "Polygon",
            "properties": {"value": "int:10"},
        }
        
        with fiona.open(
            prediction_shapefile_path,
            mode="w",
            driver="ESRI Shapefile",
            schema=pred_schema,
            crs=crs,
        ) as shp:
            for geom, value in rasterio.features.shapes(binary_predictions, mask=binary_predictions, transform=transform_pred):
                if value == 1:
                    geom_shape = shape(geom)
                    if not geom_shape.is_valid:
                        geom_shape = make_valid(geom_shape)
                    
                    if geom_shape.is_empty:
                        continue
                    
                    shp.write({"geometry": mapping(geom_shape), "properties": {"value": int(value)}})
    return inference_time, total_time
        


