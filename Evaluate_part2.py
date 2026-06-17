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
from shapely.affinity import affine_transform
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
    for modelname in ["unet_dice_ES_63.pth"]:
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

                for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                    for beta in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                        output_path = "".join([os.getcwd(), "/Test_Output/"])        
                        json_path = "".join([os.getcwd(), "/Data/JSON_files/enschede_", image[:2], ".json"])
                        label_path = "".join([output_path, "labels_", image, "_", str(alpha),"_", str(beta), ".shp"])
                        prediction_path = "".join([output_path, "prediction_", image, "_", str(alpha), "_", str(beta), ".shp"])

                        if not os.path.isdir(output_path):
                            os.makedirs(output_path)

                        polygonize_with_overlap_scores(modelname,json_path, label_path, image_index=image_index, alpha=alpha, beta=beta, prediction_shapefile_path=prediction_path)


def _normalize_image_name(image_path):
    image_name = os.path.basename(image_path)
    if not image_name.lower().endswith(".jpg"):
        image_name = f"{image_name}.jpg"
        return image_name


def get_ground_truth_labels():
    source_json = {
        "LU": os.path.join(os.getcwd(), "Data", "JSON_files", "LU.json"),
        "LR": os.path.join(os.getcwd(), "Data", "JSON_files", "LR.json"),
        "SU": os.path.join(os.getcwd(), "Data", "JSON_files", "SU.json"),
        "SR": os.path.join(os.getcwd(), "Data", "JSON_files", "SR.json")
    }

    accepted_source_images = {
        "LU": [os.path.join(os.getcwd(), "Data", "Test", "LU", "image_1"), os.path.join(os.getcwd(), "Data", "Test", "LU", "image_5"), os.path.join(os.getcwd(), "Data", "Test", "LU", "image_8"), os.path.join(os.getcwd(), "Data", "Test", "LU", "image_10"), os.path.join(os.getcwd(), "Data", "Test", "LU", "image_19")],
        "LR": [os.path.join(os.getcwd(), "Data", "Test", "LR", "image_1"), os.path.join(os.getcwd(), "Data", "Test", "LR", "image_6"), os.path.join(os.getcwd(), "Data", "Test", "LR", "image_11"), os.path.join(os.getcwd(), "Data", "Test", "LR", "image_12"), os.path.join(os.getcwd(), "Data", "Test", "LR", "image_15")],
        "SU": [os.path.join(os.getcwd(), "Data", "Test", "SU", "image_0"), os.path.join(os.getcwd(), "Data", "Test", "SU", "image_9"), os.path.join(os.getcwd(), "Data", "Test", "SU", "image_10"), os.path.join(os.getcwd(), "Data", "Test", "SU", "image_14"), os.path.join(os.getcwd(), "Data", "Test", "SU", "image_18")],
        "SR": [os.path.join(os.getcwd(), "Data", "Test", "SR", "image_1"), os.path.join(os.getcwd(), "Data", "Test", "SR", "image_5"), os.path.join(os.getcwd(), "Data", "Test", "SR", "image_10"), os.path.join(os.getcwd(), "Data", "Test", "SR", "image_12"), os.path.join(os.getcwd(), "Data", "Test", "SR", "image_16")]
    }

    numbers = {
        "LU": [0, 4, 9, 11, 18],
        "LR": [0, 7, 10, 13, 14],
        "SU": [1, 8, 11, 15, 19],
        "SR": [0, 5, 11, 13, 17]
    }

    output_dir = os.path.join(os.getcwd(), "Ground_Truth_vector")
    os.makedirs(output_dir, exist_ok=True)

    def _normalize_source_path(image_path):
        path = image_path
        if not path.lower().endswith(".jpg"):
            path = f"{path}.jpg"
        return os.path.normcase(os.path.normpath(path))

    def _normalize_annotation_path(path):
        return os.path.normcase(os.path.normpath(path))

    def _find_annotations_map(data):
        for tool_name in ["Bbox", "Rot90", "Brush"]:
            tool_data = data.get("tools_data_map", {}).get(tool_name, {})
            specifics = tool_data.get("specifics", {})
            tool_map = specifics.get(tool_name, {})
            annotations_map = tool_map.get("annotations_map")
            if isinstance(annotations_map, dict) and annotations_map:
                yield annotations_map

    def _extract_polygons(value):
        if isinstance(value, dict):
            if "Poly" in value and isinstance(value["Poly"], dict):
                points = value["Poly"].get("points", [])
                coords = [(float(p["x"]), float(p["y"])) for p in points if isinstance(p, dict) and "x" in p and "y" in p]
                if len(coords) >= 3:
                    poly = Polygon(coords)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.is_valid and not poly.is_empty:
                        yield poly
            else:
                for nested in value.values():
                    yield from _extract_polygons(nested)
        elif isinstance(value, list):
            for item in value:
                yield from _extract_polygons(item)

    def _load_georeference(context, image_index):
        geo_json = os.path.join(os.getcwd(), "Data", "JSON_files", f"enschede_{context}.json")
        if not os.path.isfile(geo_json):
            return None
        with open(geo_json, "r", encoding="utf-8") as gf:
            try:
                geo_data = json.load(gf)
            except Exception:
                return None
        if not isinstance(geo_data, list) or image_index < 0 or image_index >= len(geo_data):
            return None
        return geo_data[image_index].get("Query", {})

    def _build_transform(query):
        bbox_s = query.get("BBOX") or query.get("Bbox")
        if not bbox_s:
            return None, None
        width = int(query.get("WIDTH") or 0)
        height = int(query.get("HEIGHT") or 0)
        if width <= 0 or height <= 0:
            return None, None
        parts = re.split(r"[\,\s]+", unquote(bbox_s))
        if len(parts) < 4:
            return None, None
        minx, miny, maxx, maxy = [float(part) for part in parts[:4]]
        transform = Affine.translation(minx, maxy) * Affine.scale((maxx - minx) / width, -(maxy - miny) / height)
        crs = query.get("CRS", "")
        if isinstance(crs, str):
            crs = crs.replace("%3A", ":")
        return transform, crs

    def _project_polygon(poly, transform):
        matrix = [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]
        return affine_transform(poly, matrix)

    for context in ["LU", "LR", "SU", "SR"]:
        src_path = source_json[context]
        if not os.path.isfile(src_path):
            print(f"Source JSON not found: {src_path}")
            continue

        with open(src_path, "r", encoding="utf-8") as sf:
            try:
                src_data = json.load(sf)
            except Exception as exc:
                print(f"Failed to load {src_path}: {exc}")
                continue

        annotations_maps = list(_find_annotations_map(src_data))
        if not annotations_maps:
            print(f"No annotation maps found in {src_path}")
            continue

        for out_idx, source_path in enumerate(accepted_source_images[context], start=1):
            image_name = _normalize_image_name(source_path)
            output_name = f"{context}{out_idx}.shp"
            output_path = os.path.join(output_dir, output_name)

            annotations_key = None
            annotations_map = None
            normalized_source = _normalize_source_path(source_path)
            for annotations in annotations_maps:
                for key in annotations.keys():
                    if _normalize_annotation_path(key) == normalized_source:
                        annotations_key = key
                        annotations_map = annotations
                        break
                if annotations_key:
                    break

            if annotations_key is None or annotations_map is None:
                print(f"No annotations found for {image_name} in {src_path}")
                continue

            raw_polygons = list(_extract_polygons(annotations_map[annotations_key]))
            if not raw_polygons:
                print(f"No polygons extracted for {image_name} in {src_path}")
                continue

            if out_idx - 1 < len(numbers[context]):
                image_index = numbers[context][out_idx - 1]
            else:
                image_index = None

            if image_index is None:
                print(f"No ensechede index mapping for {image_name} in context {context}")
                continue

            query = _load_georeference(context, image_index)
            if not query:
                print(f"No georeference found for {image_name} in ensechede_{context}.json")
                continue

            transform, crs = _build_transform(query)
            if transform is None:
                print(f"Invalid georeference for {image_name} in ensechede_{context}.json")
                continue

            polygons = []
            for raw_poly in raw_polygons:
                if raw_poly.is_empty:
                    continue
                world_poly = _project_polygon(raw_poly, transform)
                if not world_poly.is_empty:
                    if not world_poly.is_valid:
                        world_poly = world_poly.buffer(0)
                    if world_poly.is_valid and not world_poly.is_empty:
                        polygons.append(world_poly)

            if not polygons:
                print(f"No projected polygons for {image_name} in {src_path}")
                continue

            gdf = gpd.GeoDataFrame(
                {"image_name": [image_name] * len(polygons), "context": [context] * len(polygons)},
                geometry=polygons,
                crs=(crs if crs else None)
            )
            try:
                gdf.to_file(output_path, driver="ESRI Shapefile")
                print(f"Saved {output_name} for {context} -> {output_path}")
            except Exception as exc:
                print(f"Failed to save {output_name}: {exc}")


def infer_metadata_from_filename(filename, accepted_source_images):
    base = os.path.splitext(filename)[0]
    context = None
    image_name = None

    match = re.match(r"^([A-Z]{2})(\d+)$", base)
    if match:
        context = match.group(1)
        idx = int(match.group(2))
        if context in accepted_source_images and 1 <= idx <= len(accepted_source_images[context]):
            image_name = _normalize_image_name(accepted_source_images[context][idx - 1])
            return context, image_name

    if "_" in base:
        prefix = base.split("_")[0]
        if prefix in accepted_source_images:
            context = prefix

    if image_name is None:
        image_name = f"{base}.jpg" if not base.lower().endswith(".jpg") else base

    return context or "", image_name

def fix_ground_truth_vector_attributes():
    output_dir = os.path.join(os.getcwd(), "Ground_Truth_vector")
    if not os.path.isdir(output_dir):
        print(f"Ground truth directory not found: {output_dir}")
        return

    accepted_source_images = {
        "LU": [os.path.join(os.getcwd(), "Data", "Test", "LU", "image_1"), os.path.join(os.getcwd(), "Data", "Test", "LU", "image_5"), os.path.join(os.getcwd(), "Data", "Test", "LU", "image_8"), os.path.join(os.getcwd(), "Data", "Test", "LU", "image_10"), os.path.join(os.getcwd(), "Data", "Test", "LU", "image_19")],
        "LR": [os.path.join(os.getcwd(), "Data", "Test", "LR", "image_1"), os.path.join(os.getcwd(), "Data", "Test", "LR", "image_6"), os.path.join(os.getcwd(), "Data", "Test", "LR", "image_11"), os.path.join(os.getcwd(), "Data", "Test", "LR", "image_12"), os.path.join(os.getcwd(), "Data", "Test", "LR", "image_15")],
        "SU": [os.path.join(os.getcwd(), "Data", "Test", "SU", "image_0"), os.path.join(os.getcwd(), "Data", "Test", "SU", "image_9"), os.path.join(os.getcwd(), "Data", "Test", "SU", "image_10"), os.path.join(os.getcwd(), "Data", "Test", "SU", "image_14"), os.path.join(os.getcwd(), "Data", "Test", "SU", "image_18")],
        "SR": [os.path.join(os.getcwd(), "Data", "Test", "SR", "image_1"), os.path.join(os.getcwd(), "Data", "Test", "SR", "image_5"), os.path.join(os.getcwd(), "Data", "Test", "SR", "image_10"), os.path.join(os.getcwd(), "Data", "Test", "SR", "image_12"), os.path.join(os.getcwd(), "Data", "Test", "SR", "image_16")]
    }

    for shp_name in sorted(os.listdir(output_dir)):
        if not shp_name.lower().endswith(".shp"):
            continue

        shp_path = os.path.join(output_dir, shp_name)
        try:
            gdf = gpd.read_file(shp_path)
        except Exception as exc:
            print(f"Failed to read {shp_name}: {exc}")
            continue

        if gdf.empty:
            print(f"Skipping empty shapefile: {shp_name}")
            continue

        context_value, image_name_value = infer_metadata_from_filename(shp_name, accepted_source_images)
        save_needed = False

        if "image_name" not in gdf.columns or gdf["image_name"].isnull().any():
            gdf["image_name"] = [image_name_value] * len(gdf)
            save_needed = True

        if "context" not in gdf.columns or gdf["context"].isnull().any():
            gdf["context"] = [context_value] * len(gdf)
            save_needed = True

        if save_needed:
            try:
                gdf.to_file(shp_path, driver="ESRI Shapefile")
                print(f"Updated {shp_name}: image_name={image_name_value}, context={context_value}")
            except Exception as exc:
                print(f"Failed to save updated {shp_name}: {exc}")


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
                        continue
                    
                    # Load ground truth labels (filtered)
                    gt_shp = os.path.join(wouter_path, f"{image_name}.shp")
                    gt_polygons = []
                    if os.path.exists(gt_shp):
                        try:
                            gt_gdf = gpd.read_file(gt_shp)
                            # Reproject GT to match original CRS if necessary
                            try:
                                if original_gdf.crs is not None and gt_gdf.crs is not None and gt_gdf.crs != original_gdf.crs:
                                    gt_gdf = gt_gdf.to_crs(original_gdf.crs)
                            except Exception as trexc:
                                continue
                            gt_polygons = list(gt_gdf.geometry)
                        except Exception as e:
                            pass
                    
                    # Load model predictions
                    model_shp = os.path.join(model_path, f"labels_{image_name}_{alpha}_{beta}.shp")
                    model_polygons = []
                    if os.path.exists(model_shp):
                        try:
                            model_gdf = gpd.read_file(model_shp)
                            model_polygons = list(model_gdf.geometry)
                        except Exception as e:
                            pass
                    
                    orig_count = len(original_gdf)
                    matched_gt_count = 0
                    matched_model_count = 0
                    matched_both_count = 0
                    matched_none_count = 0
                    
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
                            matched_both_count += 1
                        elif not in_gt and in_model:
                            fp_count += 1
                            matched_model_count += 1
                        elif in_gt and not in_model:
                            fn_count += 1
                            matched_gt_count += 1
                        elif not in_gt and not in_model:
                            tn_count += 1
                            matched_none_count += 1
                    if alpha == 0.0 and beta == 0.0:
                        print(f"summary {image_name}: original={orig_count}, gt={len(gt_polygons)}, model={len(model_polygons)}, gt_only={matched_gt_count}, model_only={matched_model_count}, both={matched_both_count}, none={matched_none_count}")

            if alpha == 0.0 and beta == 0.0:
                original_TN = tn_count
                original_TP = tp_count
                original_FP = fp_count
                original_FN = fn_count
            elif alpha == 0.0 or beta == 0.0:
                continue
            else:
                TP.append(tp_count)
                TN.append(tn_count)
                FP.append(fp_count)
                FN.append(fn_count)
                alpha_values.append(alpha)
                beta_values.append(beta)

    return TP, TN, FP, FN, alpha_values, beta_values, original_TP, original_TN, original_FN, original_FP

"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    normalize = Normalize(vmin=min(f_score), vmax=max(f_score))
    colormap = cm.get_cmap('coolwarm')  # coolwarm goes from blue to red
    colors = [colormap(normalize(value)) for value in f_score]
        
    z_values = np.zeros_like(f_score)
    ax.bar3d(alpha_values, beta_values, z_values, 0.045, 0.045, f_score, color=colors)
    ax.set_zlim(0, 1)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('F-Score')
    ax.set_title(f'F-Score for Different Alpha and Beta Values ')
    plt.show()
"""

def visualize_accuracy(TP,TN,FP,FN, alpha_values, beta_values, original_TP, original_TN, original_FN, original_FP):
    accuracy = [(TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0 for TP, TN, FP, FN in zip(TP, TN, FP, FN)]
    recall = [(TP)/(TP+FN) if (TP+FN) > 0 else 0 for TP, TN, FP, FN in zip(TP, TN, FP, FN)]
    precision = [(TP)/(TP+FP) if (TP+FP) > 0 else 0 for TP, TN, FP, FN in zip(TP, TN, FP, FN)]
    f_score = [2*(p*r)/(p+r) if (p+r) > 0 else 0 for p, r in zip(precision, recall)]

    print(f"original accuracy: {(original_TP+original_TN)/(original_TP+original_TN+original_FP+original_FN)}")
    print(f"highest accuracy: {max(accuracy)}")

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
    ax.set_title(f'Accuracy for Different Alpha and Beta Values ')
    plt.show()

    print(f"original recall: {(original_TP)/(original_TP+original_FN)}")
    print(f"highest recall: {max(recall)}")

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
    ax.set_title(f'Recall for Different Alpha and Beta Values ')
    plt.show()

    print(f"original precision: {(original_TP)/(original_TP+original_FP)}")
    print(f"highest precision: {max(precision)}")
        
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
    ax.set_title(f'Precision for Different Alpha and Beta Values ')
    plt.show()

if __name__ == "__main__":
    if not os.path.isdir(os.path.join(os.getcwd(), "Test_Output")):
        create_samples()

    gt_dir = os.path.join(os.getcwd(), "Ground_Truth_vector")
    if not os.path.isdir(gt_dir) or not any(fname.lower().endswith('.shp') for fname in os.listdir(gt_dir)):
        get_ground_truth_labels()

    orig_dir = os.path.join(os.getcwd(), "Original_labels_path")
    if not os.path.isdir(orig_dir) or not any(fname.lower().endswith('.shp') for fname in os.listdir(orig_dir)):
        get_original_labels()

    TP, TN, FP, FN, alpha_values, beta_values, original_TP, original_TN, original_FN, original_FP = calculate_accuracy()
    print("True Positives:", TP, "True Negatives:", TN, "False Positives:", FP, "False Negatives:", FN)
    visualize_accuracy(TP, TN, FP, FN, alpha_values, beta_values, original_TP, original_TN, original_FN, original_FP)
    