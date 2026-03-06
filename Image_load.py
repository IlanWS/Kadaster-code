from config import *

#import images from files into numpy arrays. 625 images of 512 by 512 arcross 3 channels (RGB)
#images are already right format, but slice anyway to ensure no shape mismatch error in the model
#before running, ensure that the data is downloaded in the correct directory, or change dir in config.py
from PIL import Image
import numpy as np
import os

def load_images():
    input_image_array = np.empty([number_of_data_pairs,512,512,3], dtype=np.int16)
    target_image_array = np.empty([number_of_data_pairs,512,512,3], dtype=np.int16)

    for i in range(number_of_data_pairs):
        #image_path1 = "".join([input_data_dir, "/image_",str(i),".jpg"])
        image_path1 = "".join([input_folder,"/image_", str(i),".jpg"])
        image1 = Image.open(image_path1)
        image_array1 = np.array(image1)
        input_image_array[i] = np.delete(np.delete(np.delete(image_array1, np.s_[512::], 0),np.s_[512::], 1),np.s_[3::], 2)

        #image_path2 = "".join([output_data_dir, "/image_", str(i),".jpg"])
        image_path2 = "".join([output_folder, "/image_", str(i), ".jpg"])
        image2 = Image.open(image_path2)
        image_array2 = np.array(image2)
        target_image_array[i] = np.delete(np.delete(np.delete(image_array2, np.s_[512::], 0),np.s_[512::], 1),np.s_[3::], 2)

#make input and output data into binary images.
#less redundent computation with minimal loss of inflormation due to data properties (near black and white images)
    input_image_array = np.mean(input_image_array, axis = 3, keepdims = True)
    input_image_array[input_image_array<128] = 0
    input_image_array[input_image_array>=128] = 1
    input_image_array = np.array(input_image_array, dtype = int)

    target_image_array = np.mean(target_image_array, axis = 3, keepdims = True)
    target_image_array[target_image_array<128] = 0
    target_image_array[target_image_array>=128] = 1
    target_image_array = np.array(target_image_array, dtype = int)

    return input_image_array, target_image_array
