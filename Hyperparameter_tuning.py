#Hyper parameter tuning. ranges here are based on a preliminary logarithmic grid search for the earning rate, and based on memory restraints for the batch size
from Model import compile_model
from config import *

import os
import matplotlib.pyplot as plt

for new_model_name in ["unet"]:
    if not os.path.isdir("".join([os.getcwd(),"/Data/Loss/", new_model_name])):
        os.makedirs("".join([os.getcwd(),"/Data/Loss/", new_model_name]))
    for hpt_lr in [0.00005, 0.00015, 0.00025, 0.00035, 0.00045]:
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

for new_model_name in ["stackedhourglass"]:
    if not os.path.isdir("".join([os.getcwd(),"/Data/Loss/", new_model_name])):
        os.makedirs("".join([os.getcwd(),"/Data/Loss/", new_model_name]))
    for hpt_lr in [0.00002, 0.00004, 0.00006, 0.00008]:
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

for new_model_name in ["deeplab"]:
    if not os.path.isdir("".join([os.getcwd(),"/Data/Loss/", new_model_name])):
        os.makedirs("".join([os.getcwd(),"/Data/Loss/", new_model_name]))
    for hpt_lr in [0.00002, 0.00004, 0.00006, 0.00008]:
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
