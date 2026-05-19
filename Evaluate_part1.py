from inference import *

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize    
from shapely.ops import unary_union

#need to be connected with server, for roadmap
def save_predictions():
    for name in ["small_scale_urban", "small_scale_rural", "large_scale_urban", "large_scale_rural"]:
        for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for beta in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                model_name = "stackedhourglass_dice_50"
                output_path = "".join([os.getcwd(), "/New_Labels/"])
                
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)

                json_path = "".join([os.getcwd(), "/Data/JSON_files/", name, ".json"])
                label_path = "".join([output_path, name, "_labels_", str(alpha),"_", str(beta), ".shp"])
                prediction_path = "".join([output_path, name, "_prediction_", str(alpha), "_", str(beta), ".shp"])

                polygonize_with_overlap_scores(json_path, label_path, image_index=0, alpha=alpha, beta=beta, prediction_shapefile_path=prediction_path)


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
    
    if merged_road_network is not None:
        for idx, row in labels_gdf.iterrows():
            label_geom = row.geometry
            total_labels_area += label_geom.area
            
            intersection = label_geom.intersection(merged_road_network)
            total_intersection_area += intersection.area
    else:
        average_overlap = 1.0
    
    # Calculate average overlap
    if total_labels_area > 0:
        average_overlap = total_intersection_area / total_labels_area
    else:
        average_overlap = 1.0
    return average_overlap

def buffer(labels_gdf, pixel_size, buffer_pixels=10):
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


#here we make a 3d barplot to visualize the average overlap for different alpha and beta values for each dataset. We iterate through each dataset, calculate the average overlap for each combination of alpha and beta, and store the results in lists. Finally, we create a 3D scatter plot to visualize the results.

def plot_results():
    for name in ["small_scale_urban", "large_scale_urban", "small_scale_rural", "large_scale_rural"]:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        output_path = "".join([os.getcwd(), "/New_Labels/"])
        json_path = "".join([os.getcwd(), "/Data/JSON_files/", name, ".json"])
        highest = 0
        alpha_values = []
        beta_values = []
        unambiguity_values = []
        legibility_values = []  


        for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for beta in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                label_path = "".join([output_path, name, "_labels_", str(alpha),"_", str(beta), ".shp"])
                merged_road_network, labels_gdf, pixel_width = get_road(json_path, label_path)
                buffered_labels_gdf = buffer(labels_gdf, pixel_size=pixel_width)

                unambiguity_score = unambiguity(merged_road_network, labels_gdf)
                legibility_score = legibility(buffered_labels_gdf)

                alpha_values.append(alpha)
                beta_values.append(beta)
                unambiguity_values.append(unambiguity_score)
                legibility_values.append(legibility_score) 


        # Create color mapping from blue (low) to red (high)
        normalize = Normalize(vmin=min(unambiguity_values), vmax=max(unambiguity_values))
        colormap = cm.get_cmap('coolwarm')  # coolwarm goes from blue to red
        colors = [colormap(normalize(value)) for value in unambiguity_values]
        
        z_values = np.zeros_like(unambiguity_values)
        ax.bar3d(alpha_values, beta_values, z_values, 0.09, 0.09, unambiguity_values, color=colors)
        ax.set_zlim(0, 1)
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_zlabel('Average Unambiguity')
        ax.set_title(f'Average Unambiguity for Different Alpha and Beta Values for {name}')
        plt.show()


if __name__ == "__main__":
    if not os.path.isdir("".join([os.getcwd(), "/New_Labels/"])):
        save_predictions()
    plot_results()
