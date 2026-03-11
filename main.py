from Model import *
from Save_prediction import *
from config import *
from visualize import *

if __name__ == '__main__':
    #train and save models
    compile_model()

    #make predictions and save them as images
    for epoch in range(10):
        name = "".join([model_name, "_", str((epoch+1)*10)])
        visualize_results(name)

    compare_predictions()
