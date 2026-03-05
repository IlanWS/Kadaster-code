from Image_load import *
from config import *

from sklearn.model_selection import train_test_split


def data_split():
    input_image_array, target_image_array = load_images()
    x_train, x_test, y_train, y_test = train_test_split(input_image_array, target_image_array, test_size=data_split_proportion, random_state=42)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test
