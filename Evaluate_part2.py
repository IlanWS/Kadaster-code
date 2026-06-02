from inference import polygonize_with_overlap_scores

import os
numbers_LU = [0, 4, 9, 11, 18]
numbers_LR = [0, 7, 10, 13, 14]
numbers_SU = [1, 8, 11, 15, 19]
numbers_SR = [0, 5, 11, 13, 17]

for modelname in ["stackedhourglass_dice_ES_73.pth", "deeplab_dice_ES_46.pth", "unet_dice_ES_63.pth"]: 
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

            if modelname.lower().startswith("deeplab"):
                alpha = 0.10
                beta = 0.60
            elif modelname.lower().startswith("stackedhourglass"):
                alpha = 0.15     
                beta = 0.50
                #alpha = 0.05
                #beta = 0.25
            elif modelname.lower().startswith("unet"):
                alpha = 0.10
                beta = 0.50

            output_path = "".join([os.getcwd(), "/Test_Output/"])        
            json_path = "".join([os.getcwd(), "/Data/JSON_files/enschede_", image[:2], ".json"])
            label_path = "".join([output_path, "labels_", image, "_", str(alpha),"_", str(beta), ".shp"])
            prediction_path = "".join([output_path, "prediction_", image, "_", str(alpha), "_", str(beta), ".shp"])

            if not os.path.isdir(output_path):
                os.makedirs(output_path)

            polygonize_with_overlap_scores(modelname,json_path, label_path, image_index=image_index, alpha=alpha, beta=beta, prediction_shapefile_path=prediction_path)

            if modelname.lower().startswith("stackedhourglass"):
                alpha = 0.05
                beta = 0.25
                label_path = "".join([output_path, "labels_", image, "_", str(alpha),"_", str(beta), ".shp"])
                prediction_path = "".join([output_path, "prediction_", image, "_", str(alpha), "_", str(beta), ".shp"])
                
                polygonize_with_overlap_scores(modelname,json_path, label_path, image_index=image_index, alpha=alpha, beta=beta, prediction_shapefile_path=prediction_path)
