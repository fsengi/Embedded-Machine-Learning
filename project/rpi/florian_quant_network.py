import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##Dataset
import os
import scipy.io as scio
from sklearn.model_selection import train_test_split

base_path = './'
dataset_path =  './Dataset' # Training data

classes = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW','PVC', 'Bigeminy', 'Trigeminy', 
           'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
ClassesNum = len(classes)

X = list()
y = list()

for root, dirs, files in os.walk(dataset_path, topdown=False):
    for name in files:
        data_train = scio.loadmat(os.path.join(root, name))# 取出字典里的value
        
        # arr -> list
        data_arr = data_train.get('val')
        data_list = data_arr.tolist()
        
        X.append(data_list[0]) # [[……]] -> [ ]
        y.append(int(os.path.basename(root)[0:2]) - 1)  # name -> num
        
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
        

X=np.array(X)
y=np.array(y)
X = X.reshape((1000,1,3600))
y = y.reshape((1000))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

batch_size = 16
class MyDataset(Dataset):
    def __init__(self):
        self.len = X_train.shape[0] # 取第0元素：长度
        self.x_train = torch.from_numpy(X_train).float()
        self.y_train = torch.from_numpy(y_train).long()
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index] # 返回对应样本即可
    def __len__(self):
        return self.len
    
class TestDataset(Dataset):
    def __init__(self):
        self.len = X_test.shape[0] # 取第0元素：长度
        self.x_test = torch.from_numpy(X_test).float()
        self.y_test = torch.from_numpy(y_test).long()
    def __getitem__(self, index):
        return self.x_test[index], self.y_test[index] # 返回对应样本即可
    def __len__(self):
        return self.len    
        
train_dataset = MyDataset()
test_dataset = TestDataset()
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=0)
test_loader = DataLoader(dataset=test_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=0)


## Model build up
num_epochs = 100
log_interval = 10
in_channels_ = 1
num_segments_in_record = 100
segment_len = 3600   # 3600 采样
num_records = 48
num_classes = 17

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class QuantizedECGNet(nn.Module):
    def __init__(self, bitwidth_matrix):
        """
        bitwidth matrix is a numpy array of shape (18,) containing the bitwidths for the different layers
        The structure is
        [Initial_act, W_1, a_1, ..., W_7, a_7, W_lin_1, a_lin_1, W_lin_2]
        - where the initial activation is an identity layer,
        - followed by 7 (Conv, ReLU, MaxPooling) blocks with W_i and a_i respectively,
        - followed by 2 Linear Layers with a ReLU inbetween.
        All biases are of Int32Bias.
        """
        super().__init__()
        self.bitwidth_matrix = bitwidth_matrix
        self.bitwidth_matrix_as_string = "_".join(map(str, bitwidth_matrix))
        print(f"This is the bitwidth matrix as a string. One layer after the other.")
        
        self.name = f"QuantECG_with{self.bitwidth_matrix_as_string}_bitwidths"

        self.network = nn.Sequential(
            qnn.QuantIdentity(bit_width=self.bitwidth_matrix[0].item(), return_quant_tensor=True),

            # 1st Layer
            qnn.QuantConv1d(1,8,16,stride=2,padding=7, bias=True, weight_bit_width=self.bitwidth_matrix[1].item(), bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=self.bitwidth_matrix[2].item(), return_quant_tensor=True),
            nn.MaxPool1d(kernel_size=8,stride=4),

            # 2nd Layer
            qnn.QuantConv1d(8,12,12,padding=5,stride=2, bias=True, weight_bit_width=self.bitwidth_matrix[3].item(), bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=self.bitwidth_matrix[4].item(), return_quant_tensor=True),
            nn.MaxPool1d(4,stride=2),
            
            # 3rd Layer
            qnn.QuantConv1d(12,32,9,stride=1,padding=4, bias=True, weight_bit_width=self.bitwidth_matrix[5].item(), bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=self.bitwidth_matrix[6].item(), return_quant_tensor=True),
            nn.MaxPool1d(5,stride=2),
            
            # 4th Layer
            qnn.QuantConv1d(32,64,7,stride=1,padding=3, bias=True, weight_bit_width=self.bitwidth_matrix[7].item(), bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=self.bitwidth_matrix[8].item(), return_quant_tensor=True),
            nn.MaxPool1d(4,stride=2),
            
            # 5th Layer
            qnn.QuantConv1d(64,64,5,stride=1,padding=2, bias=True, weight_bit_width=self.bitwidth_matrix[9].item(), bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=self.bitwidth_matrix[10].item(), return_quant_tensor=True),
            nn.MaxPool1d(2,2),
            
            # 6th Layer
            qnn.QuantConv1d(64,64,3,stride=1,padding=1, bias=True, weight_bit_width=self.bitwidth_matrix[11].item(), bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=self.bitwidth_matrix[12].item(), return_quant_tensor=True),
            nn.MaxPool1d(2,2),
            
            # 7th Layer
            qnn.QuantConv1d(64,72,3,stride=1,padding=1, bias=True, weight_bit_width=self.bitwidth_matrix[13].item(), bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=self.bitwidth_matrix[14].item(), return_quant_tensor=True),
            nn.MaxPool1d(2,2),

            # classifier
            Flatten(),
            # 1st Classification Layer
            qnn.QuantLinear(in_features=216, out_features=64, bias=True, weight_bit_width=self.bitwidth_matrix[15].item(), bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=self.bitwidth_matrix[16].item(), return_quant_tensor=True),
            nn.Dropout(p=.1),
            # 2nd Classification Layer
            qnn.QuantLinear(in_features=64, out_features=17, bias=True, weight_bit_width=self.bitwidth_matrix[17].item(), bias_quant=Int32Bias, return_quant_tensor=True),
            # why is there no softmax here?
        )

    def forward(self, x):
        x = self.network(x)
        return x
 


if __name__ == "__main__":

    correct_list = []
    model = arrhythmia_classifier()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0004, betas=(0.9, 0.999), eps=1e-08, weight_decay = 0.0, amsgrad = False)

    def train(epoch):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            optimizer.zero_grad()
            # forward + backward + update
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 10 == 9:
                # print('[%d, %5d] loss: %.8f' % (epoch + 1, batch_idx + 1, running_loss / 300))
                running_loss = 0.0


    def test():
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        correct_list.append(100 * correct / total)
        print('Accuracy on test set: %d %%' % (100 * correct / total))


    for epoch in range(50):
        train(epoch)
        test()
    print("test")
    test()

    # # Specify a path
    PATH = "florian_model_weights.pt"

    # # Save model weights
    torch.save(model.state_dict(), PATH)


    filepath = "florian_model_weights.pt"
    model = arrhythmia_classifier()

    model.load_state_dict(torch.load(filepath))
    model.eval()

    print(model)

    test()

