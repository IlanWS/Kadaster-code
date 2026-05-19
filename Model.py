from Data_preprocessing import *
from config import *
from Loss_functions import *

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights


class UNetRoadLabeler(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(UNetRoadLabeler, self).__init__()
        #this is one convolutional block, used both in the encoder and decoder
        def conv_block(in_feat, out_feat):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)
            )
        
        #encoder (downsampling)
        self.f1 = conv_block(in_channels, 64)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f2 = conv_block(64, 128)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f3 = conv_block(128, 256)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #bottleneck
        self.bottleneck = conv_block(256, 512)

        #decoder (upsampling)
        #convtranspose2d is de upsampling, daarna concatenation met de skip connection (in forard functie)
        self.u3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256) #512 because of concatenation (256+256)

        self.u2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128) 

        self.u1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64) 

        #classifier
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #encoder
        print("forward")
        f1 = self.f1(x)
        p1 = self.p1(f1)

        f2 = self.f2(p1)
        p2 = self.p2(f2)

        f3 = self.f3(p2)
        p3 = self.p3(f3)

        #bottleneck
        bn = self.bottleneck(p3)

        #decoder
        u3 = self.u3(bn)
        u3 = torch.cat([u3, f3], dim=1) #skip
        f4 = self.dec3(u3)

        u2 = self.u2(f4)
        u2 = torch.cat([u2, f2], dim=1) 
        f5 = self.dec2(u2)

        u1 = self.u1(f5)
        u1 = torch.cat([u1, f1], dim=1) 
        f6 = self.dec1(u1)

        outputs = self.final_conv(f6)
        return self.sigmoid(outputs)


class DeepLabRoadLabeler(nn.Module):
    def __init__(self, input_channels, output_channels=1, use_aux=True):
        super(DeepLabRoadLabeler, self).__init__()
        #This is a whole deeplabv3 archi with resnet101 backbone. 
        #hele deeplab archi, met ResNet backbone, ook allemaal andere versies, nog even kijken
        if os.path.isdir("".join([os.getcwd(), "/.cache/torch/hub/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth"])):
            model = DeepLabV3_ResNet101_Weights(pretrained=False)
            checkpoint = torch.load("".join([os.getcwd(), "/.cache/torch/hub/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth"]), map_location='cpu')
            model.load_state_dict(checkpoint)
            model.eval()
        else:
            weights = DeepLabV3_ResNet101_Weights.DEFAULT
            self.network = models.segmentation.deeplabv3_resnet101(weights=weights)
        

        #ResNet wants 3 channels, but we give him 1 (number of input channels so 1 for now).
        old_conv = self.network.backbone.conv1
        self.network.backbone.conv1 = nn.Conv2d(
            input_channels, 
            old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride, 
            padding=old_conv.padding, 
            bias=False
        )
        
        #averaging de pre-trained RGB weights.
        with torch.no_grad():
            self.network.backbone.conv1.weight[:] = old_conv.weight.sum(dim=1, keepdim=True)

        #DeepLab classifier is typically [Conv2d(2048, 256), Conv2d(256, num_classes)], replace just the final layer so we keep internal # of channels consistent.
        if hasattr(self.network, 'classifier') and len(self.network.classifier) >= 1:
            #if classifier is an nn.Sequential, try to replace last conv.
            if isinstance(self.network.classifier, nn.Sequential):
                last = list(self.network.classifier.children())[-1]
                if isinstance(last, nn.Conv2d):
                    in_ch = last.in_channels
                    self.network.classifier[-1] = nn.Conv2d(in_ch, output_channels, kernel_size=(1,1))
            else:
                self.network.classifier = nn.Conv2d(self.network.classifier.in_channels, output_channels, kernel_size=1)

        #use internal auxiliary arm during training to help convergence.
        if use_aux and self.network.aux_classifier is not None:
            from torchvision.models.segmentation.fcn import FCNHead
            self.network.aux_classifier = FCNHead(1024, output_channels)
        else:
            self.network.aux_classifier = None

    def forward(self, x):
        print("forward")
        input_shape = x.shape[-2:] # is (360, 640) min kan ook weg (maar niet per se met andere input configs)
        
        # output van voorgecodeerde deeplab model
        result = self.network(x)
        
        # resize output back to original input size if the stride caused a mismatch (360 is not divisible by 32)
        out = F.interpolate(result['out'], size=input_shape, mode='bilinear', align_corners=False)
        return torch.sigmoid(out)



class Residual(nn.Module):
    #smallest part of SHG model, couple of these make up one hourglass
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        if self.skip is not None:
            identity = self.skip(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)

class Hourglass(nn.Module):
    # Two of these make up the SHG model (actually num_stack, but that is 2 in this case bc of inference speed)
    def __init__(self, depth, channels):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.up1 = Residual(channels, channels)
        self.low1 = nn.MaxPool2d(2, stride=2)
        self.low2 = Residual(channels, channels)
        #Kind of like an onion, as long as depth>1, it keeps making new hourglasses inside itself 
        if self.depth > 1:
            self.low3 = Hourglass(depth - 1, channels)
        else:
            self.low3 = Residual(channels, channels)
            
        self.low4 = Residual(channels, channels)

    def forward(self, x):
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        low4 = self.low4(low3)
        # to avoid mismatches
        up2 = F.interpolate(low4, size=up1.shape[-2:], mode='nearest')
        return up1 + up2

class StackedHourglassRoadLabeler(nn.Module):
    def __init__(self, input_channels, num_stacks=2, num_channels=128): #in geval van memory overflow, doe num_channels naar 64 ofzo
        super(StackedHourglassRoadLabeler, self).__init__()
        self.num_stacks = num_stacks
        #Input channels is 1, have not tried with more yet
        #Input channels gaat uit van 1, weet niet of die werkt met iets anders tbh
        self.pre = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Residual(64, 128),
            nn.MaxPool2d(2, stride=2),
            Residual(128, 128),
            Residual(128, num_channels)
        )
        
        self.hgs = nn.ModuleList([Hourglass(4, num_channels) for i in range(num_stacks)])
        self.features = nn.ModuleList([nn.Sequential(Residual(num_channels, num_channels),
                                                    nn.Conv2d(num_channels, num_channels, 1),
                                                    nn.BatchNorm2d(num_channels),
                                                    nn.ReLU(inplace=True)) for i in range(num_stacks)])
        
        #heatmap prediction layers
        self.outs = nn.ModuleList([nn.Conv2d(num_channels, 1, 1) for i in range(num_stacks)])
        self.merge_features = nn.ModuleList([nn.Conv2d(num_channels, num_channels, 1) for i in range(num_stacks)])
        self.merge_preds = nn.ModuleList([nn.Conv2d(1, num_channels, 1) for i in range(num_stacks)])


    def forward(self, x):
        #(B, 1, 360, 640)
        print("forward")
        x = self.pre(x)
        combined_outputs = []
        
        for i in range(self.num_stacks):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_outputs.append(preds)
            
            if i < self.num_stacks - 1:
                x = x + self.merge_features[i](feature) + self.merge_preds[i](preds)
        
        #Resize final predictions to original 360x640
        final_outputs = [F.interpolate(o, size=(input_image_height, input_image_width), mode='bilinear') for o in combined_outputs]
        #Sigmoid for continuous output
        return torch.sigmoid(final_outputs[-1])



def compile_model(learning_rate=learning_rate, batch_size=batch_size, model_name=model_name, intermidiate_saves=True):
    #number of channels of the input is 1 (binary image), parameter should be changed when working with colour images (channels = 3 for RGB image)
    if model_name.lower().startswith("deeplab"):
        #result of hyperparameter tuning
        model = DeepLabRoadLabeler(1, use_aux=False)
        learning_rate = 0.00002
        batch_size = 16
    elif model_name.lower().startswith("stackedhourglass"):
        model = StackedHourglassRoadLabeler(1)
        learning_rate = 0.00004
        batch_size = 16
    elif model_name.lower().startswith("unet"):
        model = UNetRoadLabeler(1, 1)
        learning_rate = 0.0006
        batch_size = 32
    else:
        print("not a valid model name, check config.py")
        exit(1)

    #device = torch.device("xpu" if torch.xpu.is_available() else "cpu") if you use don't use NVIDIA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    #try focal cross entropy
    criterion = dice_bce_loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping()
    
    x_train, y_train, x_test, y_test = data_split()

    x_train_pt = torch.from_numpy(x_train).permute(0, 3, 1, 2).float()
    y_train_pt = torch.from_numpy(y_train).permute(0, 3, 1, 2).float()
    x_test_pt = torch.from_numpy(x_test).permute(0, 3, 1, 2).float()
    y_test_pt = torch.from_numpy(y_test).permute(0, 3, 1, 2).float()

    train_loader = DataLoader(TensorDataset(x_train_pt, y_train_pt), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_pt, y_test_pt), batch_size=batch_size, shuffle=False)

    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    print("start training")

    #start training model
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
    
    #evaluate training and validation loss in each epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, masks in test_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, masks).item()

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(test_loader):.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping at epoch:", epoch + 1)
            torch.save(early_stopping.load_best_model(model), "".join([model_path, model_name, "_", str(epoch + 1), ".pth"]))
            break

        #decided in main
        if intermidiate_saves:
            if (epoch + 1) % (epochs // 10) == 0 and epoch + 1 != epochs:
                torch.save(model.state_dict(), "".join([model_path, model_name,"_", str(epoch + 1), ".pth"]))

        if epoch == epochs - 1:
            torch.save(model.state_dict(), "".join([model_path, model_name,"_", str(epoch + 1), ".pth"]))
    #Domme pytorch heeft geen summary functie, dus je zou pytorchsummary kunnen pip installen als je wil zien, dan from pytorchsummary import summary en deze lijn:
    #summary(model, input_size=(1, 512, 512))
    return x_train, y_train, x_test, y_test




