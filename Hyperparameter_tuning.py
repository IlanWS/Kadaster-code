#Hyper parameter tuning
from Model import compile_model
from config import *

import os
import matplotlib.pyplot as plt

for new_model_name in ["deeplab","unet","stackedhourglass"]:
    if not os.path.isdir("".join([os.getcwd(),"/Data/Loss/", new_model_name])):
        os.makedirs("".join([os.getcwd(),"/Data/Loss/", new_model_name]))
    for hpt_lr in [0.001, 0.0005, 0.0001, 0.00005]:
        for hpt_bs in [8, 16, 32]:
            print(f"Running {new_model_name} training with learning rate: {hpt_lr} and batch size: {hpt_bs}")
            train_losses, val_losses = compile_model(learning_rate=hpt_lr, batch_size=hpt_bs, model_name=new_model_name)

            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.ylim(0,0.5)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Learning Curves {new_model_name} - Learning rate={hpt_lr}, Batch size={hpt_bs}')
            plt.legend()
            plt.savefig(f"Data/Loss/{new_model_name}/loss_bs_{hpt_bs}_lr_{hpt_lr}.png")
            plt.close()
