from Load_model import *
from Data_preprocessing import *
from config import *

from PIL import Image
import matplotlib.pyplot as plt
import os

def visualize_results(name):
    x_train, y_train, x_test, y_test = data_split()
    prediction = predict_model(name)

    if not os.path.isdir("".join([results_folder,model_name])):
        os.makedirs("".join([results_folder,model_name]))

    for i in range(int(number_of_data_pairs*data_split_proportion)):
        im = Image.fromarray((np.squeeze(prediction[i], axis=2)*255).astype(np.uint8))
        path = "".join([results_folder,model_name,"/",name,"/prediction_",str(i),".png"])
        im.save(path)

        im = Image.fromarray((np.squeeze(x_test[i], axis=2)*255).astype(np.uint8))
        path = "".join([results_folder,name,"/input_",str(i),".png"])
        im.save(path)
