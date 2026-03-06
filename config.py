import os
#Data specifications
json_path = "".join([os.getcwd(),"/Data"])
input_folder = "".join([os.getcwd(),"/Data/Roadnetwork"])
output_folder = "".join([os.getcwd(),"/Data/Labels"])
results_folder = "".join([os.getcwd(),"/Data/Predictions"])

number_of_data_pairs = 100
data_split_proportion=0.2

#Hyperparamers
learning_rate = 0.001
epochs = 5
batch_size = 5

#requirements
#python version 3.12.2
#numpy version 2.4.2
#torch version 2.10.0
#torchvision version 0.25.0
#pip version 25.1.1
#PIL version 12.1.1
#matplotlib version 3.10.8
#scikit-learn version 1.8.0