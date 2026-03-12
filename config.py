import os
#Data specifications

model_name = "unet_5000"
model_path = "".join([os.getcwd(),"/Models/",model_name,"/"])

json_path = "".join([os.getcwd(),"/Data/JSON_files/zoom_",model_name,".json"])
input_folder = "".join([os.getcwd(),"/Data/Roadnetwork/",model_name])
output_folder = "".join([os.getcwd(),"/Data/Labels/",model_name])
results_folder = "".join([os.getcwd(),"/Data/Predictions/"])



input_image_height = 360
input_image_width = 640
number_of_data_pairs = 900
data_split_proportion=0.2

#Hyperparamers
learning_rate = 0.001
epochs = 100
batch_size = 32