import os
#Data specifications

#chosen_model = UNetRoadLabeler(1, 1)
model_name = "zutphen_1000" #model_name should be the same as the name of the json file in the Data/JSON_files folder, and the same as the name of the folder in Data/Roadnetwork and Data/Labels, need to fix that
model_path = "".join([os.getcwd(),"/Models/",model_name,"/"])

json_path = "".join([os.getcwd(),"/Data/JSON_files/",model_name,".json"])
input_folder = "".join([os.getcwd(),"/Data/Roadnetwork/",model_name])
output_folder = "".join([os.getcwd(),"/Data/Labels/",model_name])
results_folder = "".join([os.getcwd(),"/Data/Predictions/"])

input_image_height = 360
input_image_width = 640
number_of_data_pairs = 300
data_split_proportion = 0.2

#Hyperparamers
learning_rate = 0.001
epochs = 20
batch_size = 16