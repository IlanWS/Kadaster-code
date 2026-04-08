import json
import os
import numpy as np
from PIL import Image, ImageDraw
from config import *

json_path = "".join([os.getcwd(),"/Data/JSON_files/zutphen_1000_annotation.json"])
output_dir = "".join([os.getcwd(),"/Data/Labels/zutphen_1000/"])


def extract_rvimage_masks(json_path, output_dir, width = 640, height = 360):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Navigeer naar de annotaties map
    # Structuur: tools_data_map -> Bbox -> annotations_map
    
    annotations_map = data['tools_data_map']["Bbox"]["specifics"]["Bbox"]['annotations_map']

    image_counter = 0


    for file_path, content in annotations_map.items():
        if len(content) < 2:
            continue
            
        anno_data = content[0]
        size_data = content[1]
        
        width = size_data.get('w', width)
        height = size_data.get('h', height)

        # Maak een zwart masker
        mask = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(mask)

        elements = anno_data.get('elts', [])
        has_labels = False

        for elt in elements:
            if 'Poly' in elt:
                poly_data = elt['Poly']
                
                # Haal de x,y punten op en zet ze in een lijst van tuples: [(x1, y1), (x2, y2), ...]
                if 'points' in poly_data:
                    points_list = []
                    for p in poly_data['points']:
                        points_list.append((p['x'], p['y']))
                    
                    # Teken de specifieke polygoon (driehoek, ster, etc.) in plaats van een rectangle
                    if len(points_list) >= 3:
                        draw.polygon(points_list, fill=(255, 255, 255))
                        has_labels = True

        if has_labels:
            output_name = os.path.basename(file_path).replace('.jpg', '.jpg')
            mask.save(os.path.join(output_dir, output_name), format='JPEG')
            print(f"Opgeslagen: {output_name} (Exacte vorm van {file_path})")
            image_counter += 1

extract_rvimage_masks(json_path, output_dir)

#voor de plaatjes zonder labels
for i in range(300):
    if not os.path.exists("".join([output_dir,"image_",str(i),".jpg"])):
                mask = Image.new('RGB', (input_image_width, input_image_height), (0, 0, 0))
                mask.save(os.path.join(output_dir, f"image_{i}.jpg"), format='JPEG')
                print(i)

