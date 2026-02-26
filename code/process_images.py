from PIL import Image
import numpy as np
import os

input_image_array = np.empty([625,512,512,3], dtype=np.int16)
target_image_array = np.empty([625,512,512,3], dtype=np.int16)

for i in range(625):
    image_path1 = os.path.join("zutphen-zonder-labels-map", f"image_{i}.jpg")
    image1 = Image.open(image_path1)
    image_array1 = np.array(image1)
    input_image_array[i] = np.delete(np.delete(np.delete(image_array1, np.s_[512::], 0),np.s_[512::], 1),np.s_[3::], 2)

    image_path2 = os.path.join("zutphen-met-alleen-labels-map", f"image_{i}.jpg")
    image2 = Image.open(image_path2)
    image_array2 = np.array(image2)
    target_image_array[i] = np.delete(np.delete(np.delete(image_array2, np.s_[512::], 0),np.s_[512::], 1),np.s_[3::], 2)

print("Images processed successfully")


#should make that output image is onedimentional on axis 2. ([...,512,512,1]), can turn back to image by making im[:::] = [value, value, value] 