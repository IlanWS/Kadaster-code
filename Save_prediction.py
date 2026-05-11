from Load_model import *
from Data_preprocessing import *
from config import *

from PIL import Image
import os

def visualize_results(name, x_test):
    prediction = predict_model(name, x_test)

    if not os.path.isdir("".join([results_folder,model_name,"/",name])):
        os.makedirs("".join([results_folder,model_name,"/",name]))

    for i in range(int(number_of_data_pairs*data_split_proportion)):
        im = Image.fromarray((np.squeeze(prediction[i], axis=2)*255).astype(np.uint8))
        path = "".join([results_folder,model_name,"/",name,"/prediction_",str(i),".png"])
        im.save(path)