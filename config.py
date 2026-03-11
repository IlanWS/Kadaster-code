import os
#Data specifications
json_path = "".join([os.getcwd(),"/Data/JSON_files/zoom_7500.json"])
input_folder = "".join([os.getcwd(),"/Data/Roadnetwork_7500"])
output_folder = "".join([os.getcwd(),"/Data/Labels_7500"])
results_folder = "".join([os.getcwd(),"/Data/Predictions"])

model_path = "".join([os.getcwd(),"/Models"])
model_name = "unet_7500_"

input_image_height = 360
input_image_width = 640
number_of_data_pairs = 1600
data_split_proportion=0.2

#Hyperparamers
learning_rate = 0.001
epochs = 100
batch_size = 32