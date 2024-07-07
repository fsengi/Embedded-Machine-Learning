import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ECGnet import ECGNet, TestDataset  # Ensure this import matches the actual structure

# Define the function to run inference
def run_inference(model_path, batch_size=16):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGNet().to(device)

    # Load the saved state dict
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint['net_state_dict']  # Extract only the model's state dict

    model.load_state_dict(model_state_dict)
    model.eval()  # Set model to evaluation mode

    # Prepare the test dataset
    test_dataset = TestDataset()
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Run inference
    results = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            results.extend(predicted.cpu().numpy())

    return results

if __name__ == "__main__":
    model_path = './ECGNet_model.pth'  # Update with the correct path to your model
    results = run_inference(model_path)
    print("Inference results:", results)
