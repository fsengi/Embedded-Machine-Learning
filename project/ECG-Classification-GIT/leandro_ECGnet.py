import os
import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quantization
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat

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

# Define the original model
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

# PyTorch quantization function
def quantize_model_pytorch(model):
    model_fp32 = model.to('cpu')
    model_fp32.eval()
    
    model_fp32.qconfig = quantization.get_default_qconfig('fbgemm')
    quantization.prepare(model_fp32, inplace=True)
    
    with torch.no_grad():
        for inputs, _ in train_loader:
            model_fp32(inputs)
    
    quantization.convert(model_fp32, inplace=True)
    return model_fp32

# Brevitas quantization-aware training function
class ECGNetBrevitas(nn.Module):
    def __init__(self, in_channels=1):
        super(ECGNetBrevitas, self).__init__()
        self.features = nn.Sequential(
            qnn.QuantConv1d(1, 8, 16, stride=2, padding=7, weight_quant=Int8WeightPerTensorFloat, bias=False),
            qnn.QuantReLU(act_quant=Int8ActPerTensorFloat),
            nn.MaxPool1d(kernel_size=8, stride=4),
            qnn.QuantConv1d(8, 12, 12, stride=2, padding=5, weight_quant=Int8WeightPerTensorFloat, bias=False),
            qnn.QuantReLU(act_quant=Int8ActPerTensorFloat),
            nn.MaxPool1d(kernel_size=4, stride=2),
            qnn.QuantConv1d(12, 32, 9, stride=1, padding=4, weight_quant=Int8WeightPerTensorFloat, bias=False),
            qnn.QuantReLU(act_quant=Int8ActPerTensorFloat),
            nn.MaxPool1d(kernel_size=5, stride=2),
            qnn.QuantConv1d(32, 64, 7, stride=1, padding=3, weight_quant=Int8WeightPerTensorFloat, bias=False),
            qnn.QuantReLU(act_quant=Int8ActPerTensorFloat),
            nn.MaxPool1d(kernel_size=4, stride=2),
            qnn.QuantConv1d(64, 64, 5, stride=1, padding=2, weight_quant=Int8WeightPerTensorFloat, bias=False),
            qnn.QuantReLU(act_quant=Int8ActPerTensorFloat),
            nn.MaxPool1d(kernel_size=2, stride=2),
            qnn.QuantConv1d(64, 64, 3, stride=1, padding=1, weight_quant=Int8WeightPerTensorFloat, bias=False),
            qnn.QuantReLU(act_quant=Int8ActPerTensorFloat),
            nn.MaxPool1d(kernel_size=2, stride=2),
            qnn.QuantConv1d(64, 72, 3, stride=1, padding=1, weight_quant=Int8WeightPerTensorFloat, bias=False),
            qnn.QuantReLU(act_quant=Int8ActPerTensorFloat),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(216, 64),
            qnn.QuantReLU(act_quant=Int8ActPerTensorFloat),
            nn.Dropout(p=0.1),
            nn.Linear(64, 17)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

def evaluate_model(model, val_loader, criterion, device="cuda"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader.dataset)
    val_accuracy = correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
    return val_loss, val_accuracy

# Load your dataset
dataset = MyDataset()
train_dataset = TensorDataset(dataset.x_train, dataset.y_train)
val_dataset = TensorDataset(dataset.x_test, dataset.y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Original model training and evaluation
original_model = ECGNet().to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(original_model.parameters(), lr=0.001)
train_model(original_model, train_loader, criterion, optimizer, num_epochs=25)
orig_val_loss, orig_val_acc = evaluate_model(original_model, val_loader, criterion)

# PyTorch quantization
#quantized_model_pytorch = quantize_model_pytorch(original_model)
#quantized_model_pytorch = quantized_model_pytorch.to("cuda")  # Move quantized model to CUDA
#pytorch_val_loss, pytorch_val_acc = evaluate_model(quantized_model_pytorch, val_loader, criterion, device='cuda')  # Evaluate on CUDA

# Brevitas quantization-aware training
quantized_model_brevitas = ECGNetBrevitas().to("cuda")
optimizer_brevitas = optim.Adam(quantized_model_brevitas.parameters(), lr=0.001)
train_model(quantized_model_brevitas, train_loader, criterion, optimizer_brevitas, num_epochs=25)
brevitas_val_loss, brevitas_val_acc = evaluate_model(quantized_model_brevitas, val_loader, criterion)

# Comparing the results
print(f"Original Model - Validation Loss: {orig_val_loss:.4f}, Accuracy: {orig_val_acc:.4f}")
#print(f"PyTorch Quantized Model - Validation Loss: {pytorch_val_loss:.4f}, Accuracy: {pytorch_val_acc:.4f}")
print(f"Brevitas Quantized Model - Validation Loss: {brevitas_val_loss:.4f}, Accuracy: {brevitas_val_acc:.4f}")
