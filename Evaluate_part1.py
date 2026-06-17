# In this file we conduct the first evaluation, first defining fnuctions that calculate overlap with other labels and roadpieces and such
# This is done for the whole vector layer (one image at the time) rather than per individual label. 
# We also visualize the whole lot. The evaluation depends heavily on the inferency.py file, so make sure you are running 
# Both the roadpieces (port 80, default) and labels (port 8080) on localhost in seperate ubuntu windows

from inference import *

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps as cm
from matplotlib.colors import Normalize    
from shapely.ops import unary_union

#need to be connected with server, for roadmap
def save_predictions(modelname):
    inference_times = []
    for name in ["small_scale_urban", "small_scale_rural", "large_scale_urban", "large_scale_rural"]:
        for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            for beta in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                output_path = "".join([os.getcwd(), "/New_Labels/"])
                
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)

                json_path = "".join([os.getcwd(), "/Data/JSON_files/", name, ".json"])
                label_path = "".join([output_path, name, "_labels_", str(alpha),"_", str(beta), ".shp"])
                prediction_path = "".join([output_path, name, "_prediction_", str(alpha), "_", str(beta), ".shp"])

                time, total_time = polygonize_with_overlap_scores(modelname, json_path, label_path, image_index=0, alpha=alpha, beta=beta, prediction_shapefile_path=prediction_path)
                inference_times.append((name, time, total_time))
    return inference_times

def get_road(json_path, label_shapefile_path):    
    # Load JSON to get the image URL and georeference info
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    query = data[0]['Query']
    bbox = query['BBOX'].split('%2C')
    minx, miny, maxx, maxy = map(float, bbox)
    crs = query['CRS'].replace('%3A', ':')
    
    url = data[0].get("URL")
    if not url:
        print(f"Error: No URL found in JSON")
        return None
    
    # Get road network image (on port 80)
    if url.startswith("http://localhost:8080/"):
        url = url.replace("localhost:8080/", "localhost/")
    
    road_image = download_image_from_url(url)
    if road_image is None:
        print(f"Error: Failed to download road network image")
        return None
    
    # Create binary mask of road network
    road_arr = np.array(road_image)
    
    if road_arr.ndim == 3:
        road_mask = np.mean(road_arr[:, :, :3], axis=2).astype(np.uint8)
    else:
        road_mask = road_arr
    
    if road_mask.dtype != np.uint8:
        road_mask = (road_mask * 255).astype(np.uint8)
    
    # Apply threshold to create binary mask
    binary_road_network = (road_mask > int(0.5 * 255)).astype(np.uint8)
    
    # Get image dimensions for georeference
    original_height, original_width = binary_road_network.shape
    
    # Create transform for road network
    pixel_width = (maxx - minx) / original_width
    pixel_height = (maxy - miny) / original_height
    transform_road = Affine.translation(minx, maxy) * Affine.scale(pixel_width, -pixel_height)
    
    # Extract road network geometries
    road_polygons = []
    for geom, value in rasterio.features.shapes(binary_road_network, mask=binary_road_network, transform=transform_road):
        if value == 1:
            geom_shape = shape(geom)
            if not geom_shape.is_valid:
                geom_shape = make_valid(geom_shape)
            if not geom_shape.is_empty:
                road_polygons.append(geom_shape)
    
    # Merge road network polygons
    if road_polygons:
        merged_road_network = unary_union(road_polygons)
    else:
        merged_road_network = None
    
    # Load labeled vector data from local shapefile
    labels_gdf = gpd.read_file(label_shapefile_path)
    return merged_road_network, labels_gdf, pixel_width
    
    
def unambiguity(merged_road_network, labels_gdf):
    # Calculate total areas
    total_intersection_area = 0.0
    total_labels_area = 0.0
    number_of_labels = 0

    if merged_road_network is not None:
        for idx, row in labels_gdf.iterrows():
            label_geom = row.geometry
            total_labels_area += label_geom.area
            number_of_labels += 1
            
            intersection = label_geom.intersection(merged_road_network)
            total_intersection_area += intersection.area
    else:
        return 0.0, 0
    
    # Calculate average overlap
    if total_labels_area > 0:
        average_overlap = total_intersection_area / total_labels_area
    else:
        average_overlap = 0.0
    return average_overlap, number_of_labels

def buffer(labels_gdf, pixel_size, buffer_pixels=50):
    # buffer size in map units
    buffer_distance = buffer_pixels * pixel_size

    labels_buffered = labels_gdf.copy()
    labels_buffered['geometry'] = labels_buffered.geometry.buffer(buffer_distance)
    return labels_buffered


def legibility(buffered_labels_gdf):
    # Calculate self-overlap among buffered label geometries
    total_buffer_area = buffered_labels_gdf.geometry.area.sum()
    if total_buffer_area == 0:
        return 1.0

    union_geom = unary_union(buffered_labels_gdf.geometry)
    union_area = union_geom.area
    self_overlap_area = max(total_buffer_area - union_area, 0.0)
    self_overlap_ratio = self_overlap_area / total_buffer_area
    return 1 - self_overlap_ratio

def get_number_of_labels_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    query = data[0]['Query']
    bbox = query['BBOX'].split('%2C')
    minx, miny, maxx, maxy = map(float, bbox)
    crs = query['CRS'].replace('%3A', ':')
    
    url = data[0].get("URL")
    if not url:
        print(f"Error: No URL found in JSON")
        return None
    
    # Get road network image (on port 8080)
    if url.startswith("http://localhost/"):
        url = url.replace("localhost/", "localhost:8080/")
    
    road_image = download_image_from_url(url)
    if road_image is None:
        print(f"Error: Failed to download road network image")
        return None
    
    # Create binary mask of road network
    road_arr = np.array(road_image)
    
    if road_arr.ndim == 3:
        road_mask = np.mean(road_arr[:, :, :3], axis=2).astype(np.uint8)
    else:
        road_mask = road_arr
    
    if road_mask.dtype != np.uint8:
        road_mask = (road_mask * 255).astype(np.uint8)
    
    # Apply threshold to create binary mask
    binary_labels = (road_mask > int(0.5 * 255)).astype(np.uint8)
    
    # Get image dimensions for georeference
    original_height, original_width = binary_labels.shape
    
    # Create transform for road network
    pixel_width = (maxx - minx) / original_width
    pixel_height = (maxy - miny) / original_height
    transform_road = Affine.translation(minx, maxy) * Affine.scale(pixel_width, -pixel_height)

    number_of_labels = 0
    for geom, value in rasterio.features.shapes(binary_labels, mask=binary_labels, transform=transform_road):
        if value == 1:
            number_of_labels += 1

    return number_of_labels

#here we make a 3d barplot to visualize the average overlap for different alpha and beta values for each dataset. We iterate through each dataset, calculate the average overlap for each combination of alpha and beta, and store the results in lists. Finally, we create a 3D scatter plot to visualize the results.

def plot_results(w1=1,w2=1,w3=1):
    results = {}
    original_label_scores = {}
    for name in ["small_scale_urban", "large_scale_urban", "small_scale_rural", "large_scale_rural"]:
    #for name in ["large_scale_urban"]:
        output_path = "".join([os.getcwd(), "/New_Labels/"])
        json_path = "".join([os.getcwd(), "/Data/JSON_files/", name, ".json"])
        highest = 0
        alpha_values = []
        beta_values = []
        unambiguity_values = []
        legibility_values = []
        number_of_label_values = []
        label_ratio_values = []
        label_score_values = []
        total_labels = get_number_of_labels_from_json(json_path)

        for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            for beta in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                label_path = "".join([output_path, name, "_labels_", str(alpha),"_", str(beta), ".shp"])
                merged_road_network, labels_gdf, pixel_width = get_road(json_path, label_path)
                buffered_labels_gdf = buffer(labels_gdf, pixel_size=pixel_width)
    
                unambiguity_score, number_of_labels = unambiguity(merged_road_network, labels_gdf)
                legibility_score = legibility(buffered_labels_gdf)
                label_ratio = number_of_labels/total_labels

                if alpha == 0 and beta == 0:
                    unambiguity_score_engine, number_of_labels_engine = unambiguity(merged_road_network, labels_gdf)
                    legibility_score_engine = legibility(buffered_labels_gdf)
                elif alpha == 0 or beta == 0:
                    print("")
                else: 
                    alpha_values.append(alpha)
                    beta_values.append(beta)
                    unambiguity_values.append(unambiguity_score)
                    legibility_values.append(legibility_score)
                    number_of_label_values.append(number_of_labels)
                    label_ratio_values.append(label_ratio)

                    print(f"unambiguity: {unambiguity_score:.4f}, legibility: {legibility_score:.4f}, number of labels: {number_of_labels} for alpha={alpha}, beta={beta}")
        #label_score_values = [unambiguity**2 * legibility**2 * label_ratio**2 for unambiguity, legibility, label_ratio in zip(unambiguity_values, legibility_values, label_ratio_values)]
        #label_score_values = [(w1 * (unambiguity - min(unambiguity_values)) / (max(unambiguity_values) - min(unambiguity_values)) + w2 * (legibility - min(legibility_values)) / (max(legibility_values) - min(legibility_values)) + w3 * (label_ratio - min(label_ratio_values)) / (max(label_ratio_values) - min(label_ratio_values))) / (5 * (w1 + w2 + w3)) for unambiguity, legibility, label_ratio in zip(unambiguity_values, legibility_values, label_ratio_values)]
        #label_score_values = [((unambiguity - min(unambiguity_values)) / (max(unambiguity_values) - min(unambiguity_values)))**2 * ((legibility - min(legibility_values)) / (max(legibility_values) - min(legibility_values)))**2 * ((label_ratio - min(label_ratio_values)) / (max(label_ratio_values) - min(label_ratio_values)))**2 for unambiguity, legibility, label_ratio in zip(unambiguity_values, legibility_values, label_ratio_values)]
        """
        if all(v == 0 or v == 1 for v in unambiguity_values):
            label_score_values = [((legibility - min(legibility_values)) / (max(legibility_values) - min(legibility_values)))**(1/w2) * ((label_ratio - min(label_ratio_values)) / (max(label_ratio_values) - min(label_ratio_values)))**(1/w3) for unambiguity, legibility, label_ratio in zip(unambiguity_values, legibility_values, label_ratio_values)]
        elif all(v == 0 or v == 1 for v in legibility_values):
            label_score_values = [((unambiguity - min(unambiguity_values)) / (max(unambiguity_values) - min(unambiguity_values)))**(1/w1) * ((label_ratio - min(label_ratio_values)) / (max(label_ratio_values) - min(label_ratio_values)))**(1/w3) for unambiguity, legibility, label_ratio in zip(unambiguity_values, legibility_values, label_ratio_values)]
        elif all(v == 0 or v == 1 for v in label_ratio_values):
            label_score_values = [((unambiguity - min(unambiguity_values)) / (max(unambiguity_values) - min(unambiguity_values)))**(1/w1) * ((legibility - min(legibility_values)) / (max(legibility_values) - min(legibility_values)))**(1/w2) for unambiguity, legibility, label_ratio in zip(unambiguity_values, legibility_values, label_ratio_values)]
        else:
            label_score_values = [((unambiguity - min(unambiguity_values)) / (max(unambiguity_values) - min(unambiguity_values)))**(1/w1) * ((legibility - min(legibility_values)) / (max(legibility_values) - min(legibility_values)))**(1/w2) * ((label_ratio - min(label_ratio_values)) / (max(label_ratio_values) - min(label_ratio_values)))**(1/w3) for unambiguity, legibility, label_ratio in zip(unambiguity_values, legibility_values, label_ratio_values)]
        """
        label_score_values = [((unambiguity + legibility)/2) * label_ratio**0.5 for unambiguity, legibility, label_ratio in zip(unambiguity_values, legibility_values, label_ratio_values)]

        #label_score_values = [(((unambiguity - min(unambiguity_values)) / (max(unambiguity_values) - min(unambiguity_values))) + ((legibility - min(legibility_values)) / (max(legibility_values) - min(legibility_values))))/2 + ((label_ratio - min(label_ratio_values)) / (max(label_ratio_values) - min(label_ratio_values))) for unambiguity, legibility, label_ratio in zip(unambiguity_values, legibility_values, label_ratio_values)]

        results[name] = label_score_values
        original_label_scores[name] = (unambiguity_score_engine, legibility_score_engine, number_of_labels_engine)

        # Create color mapping from blue (low) to red (high)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        normalize = Normalize(vmin=min(unambiguity_values), vmax=max(unambiguity_values))
        colormap = cm.get_cmap('coolwarm')  # coolwarm goes from blue to red
        colors = [colormap(normalize(value)) for value in unambiguity_values]
        
        z_values = np.zeros_like(unambiguity_values)
        ax.bar3d(alpha_values, beta_values, z_values, 0.045, 0.045, unambiguity_values, color=colors)
        ax.set_zlim(0, 1)
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_zlabel('Average Unambiguity')
        ax.set_title(f'Average Unambiguity for Different Alpha and Beta Values for {name}')
        plt.show()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        normalize = Normalize(vmin=min(legibility_values), vmax=max(legibility_values))
        colormap = cm.get_cmap('coolwarm')  # coolwarm goes from blue to red
        colors = [colormap(normalize(value)) for value in legibility_values]
        
        z_values = np.zeros_like(legibility_values)
        ax.bar3d(alpha_values, beta_values, z_values, 0.045, 0.045, [legibility_value-0.0 for legibility_value in legibility_values], color=colors)
        ax.set_zlim(0, 1)
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_zlabel('Average Legibility')
        ax.set_title(f'Average Legibility for Different Alpha and Beta Values for {name}')
        plt.show()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        normalize = Normalize(vmin=min(label_ratio_values), vmax=max(label_ratio_values))
        colormap = cm.get_cmap('coolwarm')  # coolwarm goes from blue to red
        colors = [colormap(normalize(value)) for value in label_ratio_values]

        print(f"number of labels: {total_labels} for {name}")
        z_values = np.zeros_like(label_ratio_values)
        ax.bar3d(alpha_values, beta_values, z_values, 0.045, 0.045, label_ratio_values, color=colors)
        ax.set_zlim(0, 1)
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_zlabel('Average Label Ratio')
        ax.set_title(f'Average Label Ratio for Different Alpha and Beta Values for {name}')
        plt.show()

        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        normalize = Normalize(vmin=min(label_score_values), vmax=max(label_score_values))
        colormap = cm.get_cmap('coolwarm')  # coolwarm goes from blue to red
        colors = [colormap(normalize(value)) for value in label_score_values]

        z_values = np.zeros_like(label_score_values)
        ax.bar3d(alpha_values, beta_values, z_values, 0.045, 0.045, label_score_values, color=colors)
        ax.set_zlim(0, 1)
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_zlabel('Average Label Score')
        ax.set_title(f'Average Label Score for Different Alpha and Beta Values for {name}')
        plt.show()
    return results, alpha_values, beta_values, original_label_scores

def plot_final_results(results, alpha_values, beta_values):
    result_small_urban = results["small_scale_urban"]
    result_small_rural = results["small_scale_rural"]
    result_large_urban = results["large_scale_urban"]
    result_large_rural = results["large_scale_rural"]

    result = [(su + sr + lu + lr) / 4 for su, sr, lu, lr in zip(result_small_urban, result_small_rural, result_large_urban, result_large_rural)]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    normalize = Normalize(vmin=min(result), vmax=max(result))
    colormap = cm.get_cmap('coolwarm')  # coolwarm goes from blue to red
    colors = [colormap(normalize(value)) for value in result]

    z_values = np.zeros_like(result)
    ax.bar3d(alpha_values, beta_values, z_values, 0.045, 0.045, result, color=colors)
    ax.set_zlim(0, 1)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Average Label Score')
    ax.set_title(f'Average Label Score for Different Alpha and Beta Values for all cases')
    plt.show()

def time_analysis(times):
    for name in ["small_scale_urban", "small_scale_rural", "large_scale_urban", "large_scale_rural"]:
        name_times = [time for n, time, total_time in times if n == name]
        name_total_times = [total_time for n, time, total_time in times if n == name]
        avg_time = sum(name_times) / len(name_times)
        avg_total_time = sum(name_total_times) / len(name_total_times)
        print(f"Average inference time for {name}: {avg_time:.4f} seconds")
        print(f"Average total time for {name}: {avg_total_time:.4f} seconds")

if __name__ == "__main__":
    times_bool = False 
    model_name = "stackedhourglass_dice_ES_73.pth"

    if not os.path.isdir("".join([os.getcwd(), "/New_Labels/"])):
        times = save_predictions(model_name)
        times_bool = True
        
    results, alpha_values, beta_values, original_label_scores = plot_results()

    plot_final_results(results,alpha_values, beta_values)

    print("time performance averages:")
    if times_bool:
        time_analysis(times)

    print("Original label scores:")
    for name, scores in original_label_scores.items(): 
        print(f"{name}: unambiguity={scores[0]:.4f}, legibility={scores[1]:.4f}, number of labels={scores[2]}") 




