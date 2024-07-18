import os
import time
import warnings
import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quantization
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.utils.prune as prune

# Suppress the deprecated warning related to TypedStorage
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Define the MyDataset class
class MyDataset(Dataset):
    def __init__(self, base_path='./', dataset_path='./Dataset', transform=None):
        self.base_path = base_path
        self.dataset_path = dataset_path
        self.classes = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW', 'PVC', 'Bigeminy',
                        'Trigeminy', 'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
        self.transform = transform

        X, y = self.load_data()
        X = self.standardize(X)
        X = X.reshape((X.shape[0], 1, 3600))  # Adjust shape if needed
        y = y.reshape((y.shape[0],))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2)
        self.len = self.X_train.shape[0]
        self.x_train = torch.from_numpy(self.X_train).float()
        self.y_train = torch.from_numpy(self.y_train).long()
        self.x_test = torch.from_numpy(self.X_test).float()
        self.y_test = torch.from_numpy(self.y_test).long()

    def load_data(self):
        X = []
        y = []
        for root, dirs, files in os.walk(self.dataset_path, topdown=False):
            for name in files:
                if name.endswith('.mat'):
                    data_train = scio.loadmat(os.path.join(root, name))
                    data_arr = data_train.get('val')
                    if data_arr is not None:
                        data_list = data_arr.tolist()
                        X.append(data_list[0])
                        y.append(int(os.path.basename(root)[0:2]) - 1)
        return np.array(X), np.array(y)

    def standardize(self, X):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.len

# Define the original model class again
class ECGNet(nn.Module):
    def __init__(self, in_channels=1):
        super(ECGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, 16, stride=2, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Conv1d(8, 12, 12, padding=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=2),
            nn.Conv1d(12, 32, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(32, 64, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=2),
            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 72, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=216, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=64, out_features=17),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.classifier(x)
        return x

# Define the quantized model
class QuantizedECGNet(nn.Module):
    def __init__(self, original_model):
        super(QuantizedECGNet, self).__init__()
        self.quant = quantization.QuantStub()
        self.features = original_model.features
        self.classifier = original_model.classifier
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

# Load the dataset (assuming the MyDataset class is defined as in your original code)
dataset = MyDataset()

# Get a single sample from the dataset
single_sample, single_label = dataset[0]
single_sample = single_sample.unsqueeze(0)  # Add batch dimension

# Function to load and run inference on a model
def run_inference(model_path, sample, device):
    model = ECGNet()
    model.load_state_dict(torch.load(model_path,  map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    sample = sample.to(device)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(sample)
        _, predicted_class = torch.max(output, 1)
    end_time = time.time()
    
    inference_time = end_time - start_time
    return predicted_class.item(), inference_time

# Function to load and run inference on a model
def run_inference_quantized(model_path, sample, device):
    original_model = ECGNet()
    qmodel = QuantizedECGNet(original_model)

    qmodel.qconfig = quantization.QConfig(
        activation=quantization.MinMaxObserver.with_args(quant_min=0, quant_max=127),
        weight=quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, quant_min=-128, quant_max=127)
    ) 
    
    quantization.prepare(qmodel, inplace=True)
    qmodel(sample)
    torch.quantization.convert(qmodel, inplace=True)

    qmodel.load_state_dict(torch.load(model_path,  map_location=torch.device('cpu')))
    qmodel.eval()  # Set the model to evaluation mode
    qmodel.to(device)
    sample = sample.to(device)
    
    start_time = time.time()
    with torch.no_grad():
        output = qmodel(sample)
        _, predicted_class = torch.max(output, 1)
    end_time = time.time()
    
    inference_time = end_time - start_time
    return predicted_class.item(), inference_time

# Load the saved models and run inference
device = torch.device('cpu')

# Original model
original_model_path = 'original_model.pth'
predicted_class_orig, inference_time_orig = run_inference(original_model_path, single_sample, device)
print(f'Original Model - Predicted class: {predicted_class_orig}, Inference time: {inference_time_orig:.6f} seconds')

# Pruned model
pruned_model_path = 'pruned_model.pth'
predicted_class_pruned, inference_time_pruned = run_inference(pruned_model_path, single_sample, device)
print(f'Pruned Model - Predicted class: {predicted_class_pruned}, Inference time: {inference_time_pruned:.6f} seconds')

# Quantized pruned model
quantized_pruned_model_path = 'quantized_pruned_model.pth'
predicted_class_quant_pruned, inference_time_quant_pruned = run_inference_quantized(quantized_pruned_model_path, single_sample, device)
print(f'Quantized Pruned Model - Predicted class: {predicted_class_quant_pruned}, Inference time: {inference_time_quant_pruned:.6f} seconds')