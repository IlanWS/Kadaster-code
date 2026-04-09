from Model import *
from Save_prediction import *
from config import *
from visualize import *

if __name__ == '__main__':
    #train and save models

    #moet de hele tijd namen van data en json aanpassen aan model naam, verandere in config wanneer we goeie data habben die voor alle model kan worden gebruikt
    compile_model()

    #make predictions and save them as images
    #callt de hele tijd data split, heel inefficient
    for epoch in range(10):
        name = "".join([model_name, "_", str((epoch+1)*(epochs//10))])
        visualize_results(name)

    compare_predictions()
