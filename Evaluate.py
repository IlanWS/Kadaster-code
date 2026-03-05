from Model import *
from Data_preprocessing import *
from config import *
from PIL import Image

import matplotlib.pyplot as plt

def visualize_results():
    x_train, y_train, x_test, y_test = data_split()
    prediction = train_model()

    print(prediction.shape)
    print(x_test.shape)
    for i in range(int(number_of_data_pairs*data_split_proportion)):
        im = Image.fromarray((np.squeeze(prediction[i], axis=2)*255).astype(np.uint8))
        path = "".join([os.getcwd(),"/Data/Predictions/prediction_",str(i),".png"])
        im.save(path)

        im = Image.fromarray((np.squeeze(x_test[i], axis=2)*255).astype(np.uint8))
        path = "".join([os.getcwd(),"/Data/Predictions/input_",str(i),".png"])
        im.save(path)

    i = 1
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(x_test[i])
    axarr[1].imshow(prediction[i])
    axarr[2].imshow(y_test[i])
    plt.show()