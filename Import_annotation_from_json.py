# Only needs to be done once, turns JSON file data from rvimage into label images through localhost map. Used to create images in Data/Labels 
# Combined data comes from different JSON files so must be ran multiple times and then run with combine_label_folders to combine

from config import *

import json
import os
import shutil
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#json_path = os.path.join(BASE_DIR, 'Data', 'JSON_files', 'deventer_2500_land_annotation.json')
#output_dir = os.path.join(BASE_DIR, 'Data', 'Roadnetwork', name) #change to "Labels" to make labels combined

output_dir = "".join([os.getcwd(), "/Ground_Truth/", "SR", "/"])
json_path = "".join([os.getcwd(), "/Data/JSON_files/", "SR.json"])


def extract_rvimage_masks(json_path, output_dir, width = 640, height = 360):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # naar annotaties map
    # tools_data_map -> Bbox -> annotations_map
    
    annotations_map = data['tools_data_map']["Bbox"]["specifics"]["Bbox"]['annotations_map']

    image_counter = 0

    for file_path, content in annotations_map.items():


        if len(content) < 2:
            print("length smaller than 2, skipping:", file_path)
            continue

        anno_data = content[0]
        size_data = content[1]

        width = size_data.get('w', width)
        height = size_data.get('h', height)

        # make empty mask
        mask = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(mask)

        elements = anno_data.get('elts', [])
        has_labels = False

        for elt in elements:
            if 'Poly' in elt:
                poly_data = elt['Poly']
                    
                    # take x and y coordinates and turn into points
                if 'points' in poly_data:
                    points_list = []
                    for p in poly_data['points']:
                        points_list.append((p['x'], p['y']))
                        
                        #make polygon
                    if len(points_list) >= 3:
                        draw.polygon(points_list, fill=(255, 255, 255))
                        has_labels = True

        if has_labels:
            output_name = os.path.basename(file_path).replace('.jpg', '.jpg')
            mask.save(os.path.join(output_dir, output_name), format='JPEG')
            print(f"Opgeslagen: {output_name} (Exacte vorm van {file_path})")
            image_counter += 1

#extract_rvimage_masks(json_path, output_dir)


def combine_label_folders(source_root, output_dir, folder_sequence, start_points=None):
    """Copy images from label subfolders into one destination folder.

    Images are renamed to image_0, image_1, ... in the combined folder.
    You can force a start index for any folder with start_points.
    """
    if start_points is None:
        start_points = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_index = 0
    for folder_name in folder_sequence:
        if folder_name in start_points:
            current_index = start_points[folder_name]

        folder_path = os.path.join(source_root, folder_name)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image_files = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )

        for image_file in image_files:
            src = os.path.join(folder_path, image_file)
            extension = os.path.splitext(image_file)[1].lower() or '.jpg'
            destination_name = f"image_{current_index}{extension}"
            dst = os.path.join(output_dir, destination_name)
            shutil.copy2(src, dst)
            print(f"Copied {folder_name}/{image_file} -> {destination_name}")
            current_index += 1

    print(f"Combined images written to: {output_dir}")


def extract():
    source_root = os.path.join(BASE_DIR, 'Data', 'Roadnetwork')
    output_dir = os.path.join(source_root, 'combined')
    folder_sequence = ['zutphen_1000','deventer_2500_land','deventer_2500_stad','apeldoorn_5000_land','apeldoorn_5000_stad']
    #numbers are the number of images in the folder, taken as index, so image names are right.
    start_points = {'zutphen_1000': 0,'deventer_2500_land': 300,'deventer_2500_stad': 525,'apeldoorn_5000_land': 750,'apeldoorn_5000_stad': 975,
                    }

    print('Combining label images into:', output_dir)
    combine_label_folders(source_root, output_dir, folder_sequence, start_points)

if __name__ == "__main__":
    #extract()
    extract_rvimage_masks(json_path, output_dir)

#To create empty mask for images without labels on them, so number of data pairs stays the same
"""
for i in range(225):
    if not os.path.exists("".join([output_dir,"image_",str(i),".jpg"])):
                mask = Image.new('RGB', (input_image_width, input_image_height), (0, 0, 0))
                mask.save(os.path.join(output_dir, f"image_{i}.jpg"), format='JPEG')
                print(i)
"""

