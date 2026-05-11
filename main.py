from Model import *
from Save_prediction import *
from config import *
from visualize import *

if __name__ == '__main__':
    #train and save models
    make_comparison = True 

    #moet de hele tijd namen van data en json aanpassen aan model naam, verandere in config wanneer we goeie data habben die voor alle model kan worden gebruikt
    x_train, y_train, x_test, y_test = compile_model(intermidiate_saves=make_comparison)

    #make predictions and save them as images
    #callt de hele tijd data split, heel inefficient
    if make_comparison:
        for epoch in range(10):
            name = "".join([model_name, "_", str((epoch+1)*(epochs//10))])
            visualize_results(name, x_test)
        compare_predictions(x_test, y_test)
