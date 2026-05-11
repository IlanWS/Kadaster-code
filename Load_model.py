from config import *
from Model import *
from Data_preprocessing import *

import torch
from torch.utils.data import DataLoader


def predict_model(name, x_test):
    
    # Must match training architecture; we trained with use_aux=False
    #model = DeepLabRoadLabeler(1, use_aux=False)
    model = StackedHourglassRoadLabeler(1)
    model.load_state_dict(torch.load("".join([model_path,name, ".pth"])))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Convert test data to tensor
    test_input_pt = torch.from_numpy(x_test).permute(0, 3, 1, 2).float()
    test_loader = DataLoader(test_input_pt, batch_size=batch_size, shuffle=False)

    # Clear GPU cache to free fragmented memory
    torch.cuda.empty_cache()

    # Process predictions in batches
    predictions_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch_predictions = model(batch)
            predictions_list.append(batch_predictions.cpu())

    # Concatenate all predictions and convert to numpy
    predictions = torch.cat(predictions_list, dim=0).numpy().transpose(0, 2, 3, 1)

    #als we binaire output willen ipv heatmap, gebruik volgende lijn.
    #predictions = (predictions > 0.5).astype(np.uint8)
    return predictions
