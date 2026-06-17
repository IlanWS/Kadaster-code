# Thought i would get real creative with it, but in the end an existing loss_function (dice_bce_loss) worked the best. womp womp
# Earlystopping is also here, both this and loss function are only called in Model.py

from config import *

import torch
from torch import nn

# Some different loss functions
class Adaptive_wing_loss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=1):
        super(Adaptive_wing_loss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        delta_y = torch.abs(y_true - y_pred)
        loss = torch.where(delta_y < self.theta,
                           self.omega * torch.log(1 + (delta_y / self.epsilon) ** self.alpha),
                           self.omega * (delta_y - self.theta) + self.omega * torch.log(torch.tensor(1 + (self.theta / self.epsilon) ** self.alpha)))
        return torch.mean(loss)
    
class dice_bce_loss(nn.Module):
    def __init__(self):
        super(dice_bce_loss, self).__init__()

    def forward(self, inputs, targets):
        # inputs are already sigmoid probabilities, so dont use BCEWithLogitsLoss
        bce = nn.BCELoss()(inputs, targets)

        #Dice Loss, inputs are probabilities
        intersection = (inputs * targets).sum()
        dice = 1 - (2. * intersection / (inputs.sum() + targets.sum() + 1e-7))  # Add epsilon to avoid division by zero
        return bce + dice
    
# Prevents overfitting
class EarlyStopping:
    def __init__(self, patience=early_stopping_patience, delta=early_stopping_delta):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)