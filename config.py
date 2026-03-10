import os
#Data specifications
json_path = "".join([os.getcwd(),"/Data/JSON_files"])
input_folder = "".join([os.getcwd(),"/Data/Roadnetwork_zoom"])
output_folder = "".join([os.getcwd(),"/Data/Labels_zoom"])
results_folder = "".join([os.getcwd(),"/Data/Predictions"])

input_image_height = 360
input_image_width = 624

number_of_data_pairs = 1200
data_split_proportion=0.2

#Hyperparamers
learning_rate = 0.001
epochs = 5
batch_size = 32