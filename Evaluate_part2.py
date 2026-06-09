from inference import polygonize_with_overlap_scores, get_labels
from Import_annotation_from_json import extract_rvimage_masks

import json
import os
import re
from urllib.parse import unquote
import geopandas as gpd
import numpy as np
import rasterio.features
from shapely.geometry import Polygon, shape
from affine import Affine
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps as cm
from matplotlib.colors import Normalize    
from shapely.ops import unary_union

def create_samples():
    numbers_LU = [0, 4, 9, 11, 18]
    numbers_LR = [0, 7, 10, 13, 14]
    numbers_SU = [1, 8, 11, 15, 19]
    numbers_SR = [0, 5, 11, 13, 17]

    #for modelname in ["stackedhourglass_dice_ES_73.pth", "deeplab_dice_ES_46.pth", "unet_dice_ES_63.pth"]: 
    for modelname in ["stackedhourglass_dice_ES_73.pth"]:
        for context in ["LU", "LR", "SU", "SR"]:
            for i in range(5):    
                image = "".join([context, str(i+1)])

                if context == "LU":
                    image_index = numbers_LU[i]
                elif context == "LR":
                    image_index = numbers_LR[i]
                elif context == "SU":
                    image_index = numbers_SU[i]
                elif context == "SR":
                    image_index = numbers_SR[i]

                """
                if modelname.lower().startswith("deeplab"):
                    alpha = 0.10
                    beta = 0.45
                elif modelname.lower().startswith("stackedhourglass"):
                    alpha = 0.25     
                    beta = 0.40
                elif modelname.lower().startswith("unet"):
                    alpha = 0.15
                    beta = 0.40
                """
                for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                    for beta in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                        output_path = "".join([os.getcwd(), "/Test_Output/"])        
                        json_path = "".join([os.getcwd(), "/Data/JSON_files/enschede_", image[:2], ".json"])
                        label_path = "".join([output_path, "labels_", image, "_", str(alpha),"_", str(beta), ".shp"])
                        prediction_path = "".join([output_path, "prediction_", image, "_", str(alpha), "_", str(beta), ".shp"])

                        if not os.path.isdir(output_path):
                            os.makedirs(output_path)

                        polygonize_with_overlap_scores(modelname,json_path, label_path, image_index=image_index, alpha=alpha, beta=beta, prediction_shapefile_path=prediction_path)

                """
                if modelname.lower().startswith("stackedhourglass"):
                    alpha = 0.10
                    beta = 0.20
                    label_path = "".join([output_path, "labels_", image, "_", str(alpha),"_", str(beta), ".shp"])
                    prediction_path = "".join([output_path, "prediction_", image, "_", str(alpha), "_", str(beta), ".shp"])
                    
                    polygonize_with_overlap_scores(modelname,json_path, label_path, image_index=image_index, alpha=alpha, beta=beta, prediction_shapefile_path=prediction_path)
                """

def get_ground_truth_labels():

    numbers_LU = [0, 4, 9, 11, 18]
    numbers_LR = [0, 7, 10, 13, 14]
    numbers_SU = [1, 8, 11, 15, 19]
    numbers_SR = [0, 5, 11, 13, 17]

    json_LU_path = os.path.join(os.getcwd(), "Data", "JSON_files", "enschede_LU.json")
    json_LR_path = os.path.join(os.getcwd(), "Data", "JSON_files", "enschede_LR.json")
    json_SU_path = os.path.join(os.getcwd(), "Data", "JSON_files", "enschede_SU.json")
    json_SR_path = os.path.join(os.getcwd(), "Data", "JSON_files", "enschede_SR.json")


    source_dirs = {
        "LR": os.path.normcase(os.path.normpath(os.path.join(os.getcwd(), "Data", "Test", "LR"))),
        "LU": os.path.normcase(os.path.normpath(os.path.join(os.getcwd(), "Data", "Test", "LU"))),
        "SR": os.path.normcase(os.path.normpath(os.path.join(os.getcwd(), "Data", "Test", "SR"))),
        "SU": os.path.normcase(os.path.normpath(os.path.join(os.getcwd(), "Data", "Test", "SU"))),
    }

    output_dir = os.path.join(os.getcwd(), "Ground_Truth_vector")
    os.makedirs(output_dir, exist_ok=True)

    for name, allowed_prefix in source_dirs.items():
        json_path = os.path.join(os.getcwd(), "Data", "JSON_files", f"{name}.json")

        if not os.path.isfile(json_path):
            print(f"JSON file not found: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotations_map = data.get("tools_data_map", {}).get("Bbox", {}).get("specifics", {}).get("Bbox", {}).get("annotations_map", {})
        if not annotations_map:
            print(f"No Bbox annotations found in {json_path}")
            continue

        # try to load matching map-request JSON (enschede_<name>.json) which contains Query BBOX/CRS
        map_json_path = os.path.join(os.getcwd(), "Data", "JSON_files", f"enschede_{name}.json")
        map_data = None
        if os.path.isfile(map_json_path):
            try:
                with open(map_json_path, "r", encoding="utf-8") as mf:
                    map_data = json.load(mf)
            except Exception:
                map_data = None

        any_saved = False
        image_count = 0
        for file_path, content in sorted(annotations_map.items()):
            normalized_path = os.path.normcase(os.path.normpath(file_path))
            if not (normalized_path == allowed_prefix or normalized_path.startswith(allowed_prefix + os.sep)):
                continue

            if not isinstance(content, list) or len(content) < 2:
                continue

            # try to determine affine transform (pixel -> map) for this image
            affine_t = None
            crs = None
            map_entry = None

            # map_data is a list where entries correspond to images; attempt to extract image index from filename
            base_name = os.path.basename(file_path)
            m = re.search(r"(\d+)", base_name)
            idx = None
            if m:
                idx = int(m.group(1)) - 1
                # Prefer Query info from the same JSON file (data) when available
                if isinstance(data, list) and idx is not None and 0 <= idx < len(data) and isinstance(data[idx], dict) and "Query" in data[idx]:
                    map_entry = data[idx]
                elif map_data and isinstance(map_data, list) and 0 <= idx < len(map_data):
                    map_entry = map_data[idx]

            # fallback: try to match by WIDTH/HEIGHT in content[1]
            img_w = None
            img_h = None
            try:
                meta = content[1]
                if isinstance(meta, dict):
                    img_w = int(meta.get("w") or meta.get("WIDTH") or 0)
                    img_h = int(meta.get("h") or meta.get("HEIGHT") or 0)
            except Exception:
                img_w = img_h = None

            if map_entry and isinstance(map_entry, dict) and "Query" in map_entry:
                q = map_entry.get("Query", {})
                bbox_s = q.get("BBOX") or q.get("Bbox")
                crs_s = q.get("CRS")
                # WIDTH/HEIGHT may be stored as strings; fall back to metadata if missing
                try:
                    width = int(q.get("WIDTH") or q.get("WIDTH") if q.get("WIDTH") else (img_w or 0))
                except Exception:
                    width = img_w or 0
                try:
                    height = int(q.get("HEIGHT") or q.get("HEIGHT") if q.get("HEIGHT") else (img_h or 0))
                except Exception:
                    height = img_h or 0

                if bbox_s and width and height:
                    # bbox in Query may be URL-encoded or comma separated
                    bbox_dec = unquote(bbox_s)
                    parts = re.split(r"[\,\s]+", bbox_dec)
                    try:
                        minx, miny, maxx, maxy = [float(p) for p in parts[:4]]
                        px = (maxx - minx) / float(width)
                        py = (maxy - miny) / float(height)
                        affine_t = Affine.translation(minx, maxy) * Affine.scale(px, -py)
                        if crs_s:
                            crs = unquote(crs_s)
                    except Exception:
                        affine_t = None

            anno_data = content[0]
            elements = anno_data.get("elts", [])
            file_records = []
            for elt in elements:
                poly_data = elt.get("Poly") if isinstance(elt, dict) else None
                if not isinstance(poly_data, dict):
                    continue

                points = poly_data.get("points", [])
                if not isinstance(points, list) or len(points) < 3:
                    continue

                coords = []
                for p in points:
                    if isinstance(p, dict) and "x" in p and "y" in p:
                        coords.append((p["x"], p["y"]))
                if len(coords) < 3:
                    continue

                # transform coords to map coords if affine available
                if affine_t is not None:
                    try:
                        world_coords = [affine_t * (float(x), float(y)) for (x, y) in coords]
                    except Exception:
                        world_coords = None
                else:
                    world_coords = None

                if world_coords:
                    poly = Polygon(world_coords)
                else:
                    poly = Polygon(coords)

                if not poly.is_valid or poly.area <= 0:
                    continue

                file_records.append({
                    "image_name": os.path.basename(file_path),
                    "image_path": file_path,
                    "geometry": poly,
                })

            if not file_records:
                continue

            gdf_file = gpd.GeoDataFrame(file_records, geometry="geometry", crs=(crs if crs else None))
            image_count += 1
            image_label = f"{name}{image_count}"
            out_shp = os.path.join(output_dir, f"{image_label}.shp")
            gdf_file.to_file(out_shp, driver="ESRI Shapefile")
            print(f"Saved {len(file_records)} polygons for {file_path} -> {out_shp}")
            any_saved = True

        if not any_saved:
            print(f"No matching polygons found for {name}")


def get_original_labels():
    numbers_LU = [0, 4, 9, 11, 18]
    numbers_LR = [0, 7, 10, 13, 14]
    numbers_SU = [1, 8, 11, 15, 19]
    numbers_SR = [0, 5, 11, 13, 17]

    output_dir = os.path.join(os.getcwd(), "Original_labels_path")
    os.makedirs(output_dir, exist_ok=True)

    for context in ["LU", "LR", "SU", "SR"]:
        if context == "LU":
            image_indices = numbers_LU
        elif context == "LR":
            image_indices = numbers_LR
        elif context == "SU":
            image_indices = numbers_SU
        elif context == "SR":
            image_indices = numbers_SR

        json_path = os.path.join(os.getcwd(), "Data", "JSON_files", f"enschede_{context}.json")

        if not os.path.isfile(json_path):
            print(f"JSON file not found: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for i, image_index in enumerate(image_indices):
            if image_index >= len(data):
                print(f"Index {image_index} out of range in {json_path}")
                continue

            image_name = f"{context}{i+1}"
            output_shp = os.path.join(output_dir, f"original_labels_{image_name}.shp")

            labels_img, label_dims = get_labels(json_path, image_index)
            if labels_img is None:
                print(f"Failed to load labels for {image_name} from {json_path}")
                continue

            query = data[image_index].get("Query", {})
            bbox_s = query.get("BBOX") or query.get("Bbox")
            crs = query.get("CRS", "").replace('%3A', ':') if query.get("CRS") else None
            width = int(query.get("WIDTH") or 0)
            height = int(query.get("HEIGHT") or 0)

            if not bbox_s or not width or not height:
                print(f"Missing georeference for {image_name} in {json_path}")
                continue

            parts = re.split(r"[\,\s]+", unquote(bbox_s))
            minx, miny, maxx, maxy = [float(v) for v in parts[:4]]
            transform = Affine.translation(minx, maxy) * Affine.scale((maxx - minx) / width, -(maxy - miny) / height)

            if labels_img.ndim == 3:
                label_mask = np.mean(labels_img[:, :, :3], axis=2).astype(np.uint8)
            else:
                label_mask = labels_img
            if label_mask.dtype != np.uint8:
                label_mask = (label_mask * 255).astype(np.uint8)
            binary_labels = (label_mask > int(0.5 * 255)).astype(np.uint8)

            polygons = []
            for geom, value in rasterio.features.shapes(binary_labels, mask=binary_labels, transform=transform):
                if value != 1:
                    continue
                poly = shape(geom)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    polygons.append(poly)

            if not polygons:
                print(f"No label polygons found for {image_name}")
                continue

            gdf = gpd.GeoDataFrame({"image_name": [image_name] * len(polygons)}, geometry=polygons, crs=(crs if crs else None))
            gdf.to_file(output_shp, driver="ESRI Shapefile")
            print(f"Saved original labels for {image_name} -> {output_shp}")


def calculate_accuracy():
    wouter_path = os.path.join(os.getcwd(), "Ground_Truth_vector")
    original_path = os.path.join(os.getcwd(), "Original_labels_path")
    model_path = os.path.join(os.getcwd(), "Test_Output")
    TP = []
    TN = []
    FP = []
    FN = []
    alpha_values = []
    beta_values = []

    contexts = ["LR", "LU", "SR", "SU"]
    images_per_context = 5
    
    # Define overlap threshold for matching polygons
    overlap_threshold = 0.1  # 10% intersection over union

    for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for beta in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            print(f"Evaluating for alpha={alpha}, beta={beta}")
            tp_count = 0
            tn_count = 0
            fp_count = 0
            fn_count = 0
            
            for context in contexts:
                for i in range(1, images_per_context + 1):
                    image_name = f"{context}{i}"
                    
                    # Load original labels
                    original_shp = os.path.join(original_path, f"original_labels_{image_name}.shp")
                    if not os.path.exists(original_shp):
                        continue
                    try:
                        original_gdf = gpd.read_file(original_shp)
                    except Exception as e:
                        print(f"Error reading {original_shp}: {e}")
                        continue
                    
                    # Load ground truth labels (filtered)
                    gt_shp = os.path.join(wouter_path, f"{image_name}.shp")
                    gt_polygons = []
                    if os.path.exists(gt_shp):
                        try:
                            gt_gdf = gpd.read_file(gt_shp)
                            gt_polygons = list(gt_gdf.geometry)
                        except Exception:
                            pass
                    
                    # Load model predictions
                    model_shp = os.path.join(model_path, f"labels_{image_name}_{alpha}_{beta}.shp")
                    model_polygons = []
                    if os.path.exists(model_shp):
                        try:
                            model_gdf = gpd.read_file(model_shp)
                            model_polygons = list(model_gdf.geometry)
                        except Exception:
                            pass
                    
                    # Evaluate each original label
                    for orig_poly in original_gdf.geometry:
                        if not orig_poly.is_valid:
                            orig_poly = orig_poly.buffer(0)
                        if orig_poly.is_empty:
                            continue
                        
                        # Check if this original label is in ground truth (positive case)
                        in_gt = False
                        for gt_poly in gt_polygons:
                            if not gt_poly.is_valid:
                                gt_poly = gt_poly.buffer(0)
                            intersection = orig_poly.intersection(gt_poly)
                            union = orig_poly.union(gt_poly)
                            if union.area > 0:
                                iou = intersection.area / union.area
                                if iou > overlap_threshold:
                                    in_gt = True
                                    break
                        
                        # Check if this original label is in model prediction
                        in_model = False
                        for model_poly in model_polygons:
                            if not model_poly.is_valid:
                                model_poly = model_poly.buffer(0)
                            intersection = orig_poly.intersection(model_poly)
                            union = orig_poly.union(model_poly)
                            if union.area > 0:
                                iou = intersection.area / union.area
                                if iou > overlap_threshold:
                                    in_model = True
                                    break
                        
                        # Classify
                        if in_gt and in_model:
                            tp_count += 1
                        elif not in_gt and in_model:
                            fp_count += 1
                        elif in_gt and not in_model:
                            fn_count += 1
                        elif not in_gt and not in_model:
                            tn_count += 1
            
            TP.append(tp_count)
            TN.append(tn_count)
            FP.append(fp_count)
            FN.append(fn_count)
            alpha_values.append(alpha)
            beta_values.append(beta)

    return TP, TN, FP, FN, alpha_values, beta_values

def visualize_accuracy(TP,TN,FP,FN, alpha_values, beta_values):
    accuracy = [(TP+TN)/(TP+TN+FP+FN) for TP, TN, FP, FN in zip(TP, TN, FP, FN)]
    recall = [(TP)/(TP+FN) for TP, TN, FP, FN in zip(TP, TN, FP, FN)]
    precision = [(TP)/(TP+FP) for TP, TN, FP, FN in zip(TP, TN, FP, FN)]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    normalize = Normalize(vmin=min(accuracy), vmax=max(accuracy))
    colormap = cm.get_cmap('coolwarm')  # coolwarm goes from blue to red
    colors = [colormap(normalize(value)) for value in accuracy]
        
    z_values = np.zeros_like(accuracy)
    ax.bar3d(alpha_values, beta_values, z_values, 0.045, 0.045, accuracy, color=colors)
    ax.set_zlim(0, 1)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Accuracy')
    ax.set_title(f'Accuracy for Different Alpha and Beta Values (cartographer)')
    plt.show()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    normalize = Normalize(vmin=min(recall), vmax=max(recall))
    colormap = cm.get_cmap('coolwarm')  # coolwarm goes from blue to red
    colors = [colormap(normalize(value)) for value in recall]
        
    z_values = np.zeros_like(recall)
    ax.bar3d(alpha_values, beta_values, z_values, 0.045, 0.045, recall, color=colors)
    ax.set_zlim(0, 1)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Recall')
    ax.set_title(f'Recall for Different Alpha and Beta Values (cartographer)')
    plt.show()
        
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    normalize = Normalize(vmin=min(precision), vmax=max(precision))
    colormap = cm.get_cmap('coolwarm')  # coolwarm goes from blue to red
    colors = [colormap(normalize(value)) for value in precision]
        
    z_values = np.zeros_like(precision)
    ax.bar3d(alpha_values, beta_values, z_values, 0.045, 0.045, precision, color=colors)
    ax.set_zlim(0, 1)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Precision')
    ax.set_title(f'Precision for Different Alpha and Beta Values (cartographer)')
    plt.show()

if __name__ == "__main__":
    if not os.path.isdir("".join([os.getcwd(), "/Test_Output/"])):
        create_samples()
    if not os.path.isdir("".join([os.getcwd(), "/Ground_Truth_vector/"])):
        get_ground_truth_labels()
    if not os.path.isdir("".join([os.getcwd(), "/Original_labels_path/"])):
        get_original_labels()
    TP, TN, FP, FN, alpha_values, beta_values = calculate_accuracy()
    print("True Positives:", TP, "True Negatives:", TN, "False Positives:", FP, "False Negatives:", FN)
    visualize_accuracy(TP, TN, FP, FN, alpha_values, beta_values)



