from Data_preprocessing import *
from config import *
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


def compare_predictions():
    x_train, y_train, x_test, y_test = data_split()

    if not os.path.isdir("".join([results_folder,"/Comparisons"])):
        os.makedirs("".join([results_folder,"/Comparisons"]))

    for i in range(1):
        f, axarr = plt.subplots(3, 4)
        axarr[0, 0].imshow(x_test[i])
        axarr[0, 1].imshow(y_test[i])
        axarr[0, 2].imshow(np.array(Image.open("".join([results_folder, model_name, "_10/prediction_",str(i),".png"]))))
        axarr[0, 3].imshow(np.array(Image.open("".join([results_folder, model_name, "_20/prediction_",str(i),".png"]))))
        axarr[1, 0].imshow(np.array(Image.open("".join([results_folder, model_name, "_30/prediction_",str(i),".png"]))))
        axarr[1, 1].imshow(np.array(Image.open("".join([results_folder, model_name, "_40/prediction_",str(i),".png"]))))
        axarr[1, 2].imshow(np.array(Image.open("".join([results_folder, model_name, "_50/prediction_",str(i),".png"]))))
        axarr[1, 3].imshow(np.array(Image.open("".join([results_folder, model_name, "_60/prediction_",str(i),".png"]))))
        axarr[2, 0].imshow(np.array(Image.open("".join([results_folder, model_name, "_70/prediction_",str(i),".png"]))))
        axarr[2, 1].imshow(np.array(Image.open("".join([results_folder, model_name, "_80/prediction_",str(i),".png"]))))
        axarr[2, 2].imshow(np.array(Image.open("".join([results_folder, model_name, "_90/prediction_",str(i),".png"]))))
        axarr[2, 3].imshow(np.array(Image.open("".join([results_folder, model_name,"_100/prediction_",str(i),".png"]))))
        plt.savefig("".join([results_folder,"/Comparisons/compare_",str(i),".png"]))