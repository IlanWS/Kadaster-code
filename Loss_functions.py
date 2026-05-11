# some different lossfunction (make own class in .py file to call in model .py file)
import torch
from torch import nn

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
        # BCE Loss - inputs are already sigmoid probabilities, so use BCELoss instead of BCEWithLogitsLoss
        bce = nn.BCELoss()(inputs, targets)

        # Dice Loss - inputs are already probabilities
        probs = inputs  # No need to apply sigmoid again
        intersection = (probs * targets).sum()
        dice = 1 - (2. * intersection / (probs.sum() + targets.sum() + 1e-7))  # Add epsilon to avoid division by zero
        return bce + dice