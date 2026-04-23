# polygonize inference results from a raster mask and write vector polygons with rasterio
import numpy as np
from PIL import Image
from affine import Affine
from Load_model import predict_model
import rasterio.features
import fiona
from shapely.geometry import shape, mapping
import json
import matplotlib.pyplot as plt

prediction_path = "C:\\Users\\SmeerdijkIlan\\Documents\\Master_thesis_opdracht\\Data\\Predictions\\StackedHourglass_combined\\StackedHourglass_combined_100\\prediction_0.png" 
#prediction_path = ("C:\\Users\\SmeerdijkIlan\\Documents\\Master_thesis_opdracht\\Data\\Labels\\amsterdam\\image_0.jpg")
json_path = "C:\\Users\\SmeerdijkIlan\\Documents\\Master_thesis_opdracht\\Data\\JSON_files\\amsterdam.json"
shapefile_path = "polygonized_prediction.shp"

def polygonize_results(prediction_path, shapefile_path, json_path, threshold=0.5):
    # Load JSON for georeferencing info
    with open(json_path, 'r') as f:
        data = json.load(f)
    query = data[0]['Query']
    bbox = query['BBOX'].split('%2C')
    minx, miny, maxx, maxy = map(float, bbox)
    crs = query['CRS'].replace('%3A', ':')
    width = int(query['WIDTH'])
    height = int(query['HEIGHT'])

    img = Image.open(prediction_path)
    print(type(img), img)
    arr = np.array(img)
    print(type(arr), arr.shape, arr)

    plt.imshow(arr.squeeze(), cmap='gray')
    plt.show()

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

    schema = {
        "geometry": "Polygon",
        "properties": {"value": "int"},
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
                shp.write({"geometry": geom, "properties": {"value": int(value)}})



polygonize_results(prediction_path, shapefile_path, json_path, threshold=0.5)

