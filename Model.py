from Data_preprocessing import *
from Hyperparameters import *

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class UNetRoadLabeler(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(UNetRoadLabeler, self).__init__()
        def conv_block(in_feat, out_feat):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)
            )
        # Encoder (Downsampling)
        self.f1 = conv_block(in_channels, 64)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f2 = conv_block(64, 128)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f3 = conv_block(128, 256)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = conv_block(256, 512)

        # Decoder (Upsampling)
        self.u3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256) # 512 because of concatenation (256+256)

        self.u2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128) # 128+128

        self.u1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)   # 64+64

        # Final Output Layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        f1 = self.f1(x)
        p1 = self.p1(f1)

        f2 = self.f2(p1)
        p2 = self.p2(f2)

        f3 = self.f3(p2)
        p3 = self.p3(f3)

        # Bottleneck
        bn = self.bottleneck(p3)

        # Decoder
        u3 = self.u3(bn)
        u3 = torch.cat([u3, f3], dim=1) # Skip connection
        f4 = self.dec3(u3)

        u2 = self.u2(f4)
        u2 = torch.cat([u2, f2], dim=1) # Skip connection
        f5 = self.dec2(u2)

        u1 = self.u1(f5)
        u1 = torch.cat([u1, f1], dim=1) # Skip connection
        f6 = self.dec1(u1)

        outputs = self.final_conv(f6)
        return self.sigmoid(outputs)

def compile_model():
    #number of channels of the input is 1 (binary image), parameter should be changed when working with colour images (channels = 3 for RGB image)
    model = UNetRoadLabeler(1,1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #try focal cross entropy
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    x_train, y_train, x_test, y_test = data_split()

    x_train_pt = torch.from_numpy(x_train).permute(0, 3, 1, 2).float()
    y_train_pt = torch.from_numpy(y_train).permute(0, 3, 1, 2).float()
    x_test_pt = torch.from_numpy(x_test).permute(0, 3, 1, 2).float()
    y_test_pt = torch.from_numpy(y_test).permute(0, 3, 1, 2).float()

    train_loader = DataLoader(TensorDataset(x_train_pt, y_train_pt), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_pt, y_test_pt), batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, masks in train_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, masks in test_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, masks).item()

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(test_loader):.4f}")

    #Domme pytorch heeft geen summary functie, dus je zou pytorchsummary kunnen pip installen als je wil zien, dan from pytorchsummary import summary en deze lijn:
    #summary(model, input_size=(1, 512, 512))
    return model


def train_model():
    model = compile_model()
    x_train, y_train, x_test, y_test = data_split()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_input = torch.from_numpy(x_test).permute(0, 3, 1, 2).float().to(device)

    with torch.no_grad():
        predictions = model(test_input)

    predictions = predictions.cpu().numpy().transpose(0, 2, 3, 1)

    #als we binaire output willen ipv heatmap, gebruik volgende lijn.
    #predictions = (predictions > 0.5).astype(np.uint8)
    return predictions

