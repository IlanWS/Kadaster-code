from Model import *
from Save_prediction import *

if __name__ == '__main__':
    #train and save models
    compile_model()

    #make predictions and save them as images
    for epoch in range(10):
        model_name = "".join(["unet_", str((epoch+1)*10)])
        visualize_results(model_name)