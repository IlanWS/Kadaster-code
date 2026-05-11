from Data_preprocessing import *
from config import *

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


def compare_predictions(x_test, y_test):
    if not os.path.isdir("".join([results_folder,model_name,"/Comparisons"])):
        os.makedirs("".join([results_folder,model_name,"/Comparisons"]))

    for i in range(11):
        plt.subplot(3, 4, 1)
        plt.imshow(x_test[i])
        plt.title("Input image")
        plt.subplot(3, 4, 2)
        plt.imshow(y_test[i])
        plt.title("Ground truth")
        for epoch in range(10):
            name = "".join([model_name, "_", str((epoch + 1) * (epochs // 10))])
            plt.subplot(3, 4, epoch + 3)
            plt.imshow(np.array(Image.open("".join([results_folder, model_name,"/", name,"/prediction_",str(i),".png"]))))
            plt.title("Epochs: " + str((epoch + 1) * (epochs // 10)))
        plt.savefig("".join([results_folder,model_name,"/Comparisons/compare_",str(i),".png"]))
        plt.clf()

