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
import torch.nn.utils.prune as prune

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


# PyTorch quantization function
def quantize_model_pytorch(model, train_loader):
    model_fp32 = QuantizedECGNet(model).to('cpu')
    model_fp32.eval()
    
    model_fp32.qconfig = quantization.QConfig(
        activation=quantization.MinMaxObserver.with_args(quant_min=0, quant_max=127),
        weight=quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, quant_min=-128, quant_max=127)
    )

    quantization.prepare(model_fp32, inplace=True)
   
    with torch.no_grad():
        for inputs, _ in train_loader:
            model_fp32(inputs)
    
    quantization.convert(model_fp32, inplace=True)
    return model_fp32

# Global pruning function
def global_prune_model(model, amount=0.5):
    # Collect all the parameters to prune
    parameters_to_prune = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    
    # Remove the pruning reparameterization
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    
    return model

def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            #inputs, labels = inputs.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

def evaluate_model(model, val_loader, criterion, device="cpu"):
    model.to(device)
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
original_model = ECGNet().to("cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(original_model.parameters(), lr=0.001)
print(f"\n---Training original model, 250 epochs")
train_model(original_model, train_loader, criterion, optimizer, num_epochs=250)
orig_val_loss, orig_val_acc = evaluate_model(original_model, val_loader, criterion, device="cpu")
torch.save(original_model.state_dict(), 'original_model.pth')

# Optional: Apply global pruning
pruned_model = global_prune_model(original_model, amount=0.5)
# Re-train pruned model
print(f"\n---Reraining pruned model, 50 epochs")
train_model(pruned_model, train_loader, criterion, optimizer, num_epochs=50)
pruned_val_loss, pruned_val_acc = evaluate_model(pruned_model, val_loader, criterion, device="cpu")
torch.save(pruned_model.state_dict(), 'pruned_model.pth')

# Quantize the pruned model using PyTorch
print(f"\n---Quantizing pruned model")
quantized_pruned_model_pytorch = quantize_model_pytorch(pruned_model, train_loader)
# Evaluate the quantized pruned model
quantized_pruned_val_loss, quantized_pruned_val_acc = evaluate_model(quantized_pruned_model_pytorch, val_loader, criterion, device="cpu")  # Evaluate on CPU
torch.save(quantized_pruned_model_pytorch.state_dict(), 'quantized_pruned_model.pth')

# Comparing the results
print(f"\n---Final results:\n")
print(f"Original Model - Validation Loss: {orig_val_loss:.4f}, Accuracy: {orig_val_acc:.4f}")
print(f"Pruned Model - Validation Loss: {pruned_val_loss:.4f}, Accuracy: {pruned_val_acc:.4f}")
print(f"Quantized Pruned Model - Validation Loss: {quantized_pruned_val_loss:.4f}, Accuracy: {quantized_pruned_val_acc:.4f}")
