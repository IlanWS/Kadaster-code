import matplotlib.pyplot as plt
from Model import *
from Data_preprocessing import data_split

def visualize_results():
    x_train, y_train, x_test, y_test = data_split()
    prediction = train_model()

    i = 1
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(x_test[i])
    axarr[1].imshow(prediction[i])
    axarr[2].imshow(y_test[i])
    plt.show()

