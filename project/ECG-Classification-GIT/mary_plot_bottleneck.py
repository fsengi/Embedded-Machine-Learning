import argparse
import os # load data, models,...
import time # time training
from datetime import datetime
import json # log for training
import re

# torch imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchsummary import summary # nice displays of networks

# data preparation and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.model_selection import train_test_split


# Defining the network (ECGNet5)  
in_channels_ = 1
num_segments_in_record = 100
segment_len = 3600
num_classes = 17
classes = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW','PVC', 'Bigeminy', 'Trigeminy', 
    'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
ClassesNum = len(classes)
        
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

class MyDataset(Dataset):
    def __init__(self, X,y):
        self.len = X.shape[0] 
        self.x = torch.from_numpy(X).float().to("cuda")
        self.y = torch.from_numpy(y).long().to("cuda")
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
# Why don't they use the Batch Normalization? 


class ECGNet(nn.Module):
    """ECGNet from paper with additional Identity layer"""
    def __init__(self):
        super().__init__()
        self.name = "ECGNet"

        self.network = nn.Sequential(
            # 1st conv block       
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=16,stride=2,padding=7, bias=False), # index: 0
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8,stride=4),
            # 2nd conv block
            nn.Conv1d(8,12,12,padding=5,stride=2, bias=False), # index: 4
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.MaxPool1d(4,stride=2),
            # 3rd
            nn.Conv1d(12,32,9,stride=1,padding=4, bias=False), # 8
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(5,stride=2),
            # 4th
            nn.Conv1d(32,64,7,stride=1,padding=3, bias=False), # 12
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4,stride=2),
            # 5th
            nn.Conv1d(64,64,5,stride=1,padding=2, bias=False), # 16
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2,2),
            # 6th
            nn.Conv1d(64,64,3,stride=1,padding=1, bias=False), # 20
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2,2),
            # 7th
            nn.Conv1d(64,72,3,stride=1,padding=1, bias=False), # 24
            nn.BatchNorm1d(72),
            nn.ReLU(),
            nn.MaxPool1d(2,2),
        )
        self.classifier=nn.Sequential(
            # classifier
            nn.Linear(in_features=216, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(in_features=64, out_features=17),
        )
    def forward(self, x):
        x = self.network(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, kernel_size, padding):
        super(Bottleneck, self).__init__()
        #print(f'in_channels: {in_channels}, bottleneck_channels: {bottleneck_channels}, out_channels: {out_channels}')
        self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x)) # remove non-linearities in bottleneck block?
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class ECGNetWithBottleneckLayer(nn.Module):
    def __init__(self, original_model, n, bottleneck_fraction): # bottleneck_channels as parameter?
        """This class takes a trained ECGNet and replaces one convolutional layer with a BottleneckLayer.
        
        Args:
        original_model: an ECGNet instance (could in principal be another model of similar structure)
        n: the n-th convolutional layer will be replaced; values in [0,...,6]
        new_layer: the new_layer to be inserted
        bottleneck_fraction: how much the layer will be reduced. 3 Versions: [0.75, 0.5, 0.25]

        """
        super(ECGNetWithBottleneckLayer, self).__init__()
        self.name = f"ECGNetWithBottleneck_{n}_bottleneckfrac_{[0.75,0.5,0.25].index(bottleneck_fraction)}"
        #print(f"defined {self.name}")
        self.layers = nn.ModuleList()
        self.n = n
        self.bottleneck_channels = [
            int(original_model.network[4*n].in_channels),
            int(bottleneck_fraction*(min(int(original_model.network[4*n].in_channels), int(original_model.network[4*n].out_channels)))),
            int(original_model.network[4*n].out_channels),
            ]
        self.fc_input_shapes = [432,216,216,216,216,216]

        new_layer = Bottleneck(
            in_channels=self.bottleneck_channels[0],
            bottleneck_channels=self.bottleneck_channels[1],
            out_channels=self.bottleneck_channels[2],
            kernel_size=original_model.network[4*n].kernel_size,
            padding=original_model.network[4*n].padding,
            )
        
        for i in range(4*n): # copy layers 0 to n (blocks are of size 4: Conv,BatchNorm,ReLU,MaxPool)
            self.layers.append(original_model.network[i])
        
        self.layers.append(new_layer)
        
        for i in range(4*n+1, len(original_model.network)): # copy the remaining layers
            self.layers.append(original_model.network[i])
        self.classifier=nn.Sequential(
            # classifier
            nn.Linear(in_features=self.fc_input_shapes[n-1], out_features=64),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(in_features=64, out_features=17),
        )
    
    def calculate_length_after_conv(input_length, kernel_size, stride=1, padding=0, dilation=1):
        return (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            #print(f'Shape after layer: {x.shape}')
        
        x = x.view(x.size(0), -1) 

        #print(f'Shape before fc: {x.shape}')
        x = self.classifier(x)
        return x


def prepare_data(batch_size):
    base_path = './'
    dataset_path =  './Dataset'

    X = list()
    y = list()

    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            data_train = scio.loadmat(os.path.join(root, name))
            
            # arr -> list
            data_arr = data_train.get('val')
            data_list = data_arr.tolist()
            
            X.append(data_list[0]) # [[……]] -> [ ]
            y.append(int(os.path.basename(root)[0:2]) - 1)

    X=np.array(X)
    y=np.array(y)
    X = standardization(X)
    X = X.reshape((1000,1,3600))
    y = y.reshape((1000))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("X_train : ", len(X_train))
    print("X_test  : ", len(X_test))
    print("shape of X_train : ", np.shape(X_train[0]))
    print("shape of y_train : ", np.shape(y_train))
    print("shape of X_test : ", np.shape(X_test))
    print("shape of y_test : ", np.shape(y_test))
            
    train_dataset = MyDataset(X=X_train,y=y_train)
    test_dataset = MyDataset(X=X_test,y=y_test)
    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=0)
    entire_test_set_loader = DataLoader(dataset=test_dataset, 
                            batch_size=200,
                            shuffle=True, 
                            num_workers=0)
    return train_loader, test_loader, entire_test_set_loader
   

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    #print(f"Current time: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}; Test Epoch: {epoch}, Test set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc}%)\n")
    return test_loss, acc

def plot_all_models_test_acc(data:dict) -> None:
    # TODO: 4 colours?
    plt.figure()
    for key in data.keys():
        # plot 
        _data = data[key]
        key = str(key).replace("ECGNetWithBottleneck","Bottleneck")

        # TODO introduce if frac==0 label = Bottleneck layer that is replaced = "" as key via string operations
        plt.plot(_data['accTest'], '-', label=key)
    
    plt.title("Test accuracy of all models over 30 epochs")
    plt.legend(loc=(1.04, 0))
    plt.show()
    plt.savefig("all_models.png", bbox_inches="tight")

def remove_bottleneck_pattern(s):
    # Regular expression to match '_bottleneck_' followed by one or more digits
    pattern = re.compile(r'_bottleneckfrac_\d+')
    # Use sub to replace the pattern with an empty string
    return pattern.sub('', s)

def plot_test_acc_for_specific_frac(data:dict,fraction=None) -> None:
    plt.figure()
    for key in data.keys():
        # plot 
        _data = data[key]
        key = str(key).replace("ECGNetWithBottleneck","Bottleneck")
        print(remove_bottleneck_pattern(key))
        # TODO introduce if frac==0 label = Bottleneck layer that is replaced = "" as key via string operations
        plt.plot(_data['accTest'], '-', label=remove_bottleneck_pattern(key).replace("_", " Layer = "))
    plt.ylabel("Test accuracy")
    plt.xlabel("Epochs")
    if fraction==0:
        plt.title("Test accuracies of models with 0.75 bottleneck fraction")
        plt.legend(loc=(1.04, 0))
        plt.show()
        plt.savefig("all_models_with_bottleneck_frac_0.png", bbox_inches="tight")
    elif fraction==1:
        plt.title("Test accuracies of models with 0.5 bottleneck fraction")
        plt.legend(loc=(1.04, 0))
        plt.show()
        plt.savefig("all_models_with_bottleneck_frac_1.png", bbox_inches="tight")
    elif fraction==2:
        plt.title("Test accuracies of models with 0.25 bottleneck fraction")
        plt.legend(loc=(1.04, 0))
        plt.show()
        plt.savefig("all_models_with_bottleneck_frac_2.png", bbox_inches="tight")

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='Log_interval',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--bit-widths', type=int, default=4, metavar='bitwidth', nargs='+',
                        help='how many bits for quantization')
    parser.add_argument('--load-model', type=bool, default=False, metavar='load_pretrained_models')
    parser.add_argument('--train-log-path', type=str, default="mary_trained_networks.json", help='a .json file where you want to store the train data')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # reproducibility and ensure train-test separation 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device

    train_loader, test_loader, entire_test_loader = prepare_data(batch_size=args.batch_size)

    models = []

    # load classical model
    ecg_net = ECGNet().cuda()
    # TODO actually load model

    model_path = "ECGNet_trained_500_epochs.pth"
    ecg_net.load_state_dict(torch.load(model_path))

    _, test_acc = test(ecg_net, device, test_loader=entire_test_loader, epoch=499)
    print(f"ecg et has accuracy {test_acc}")

    #summary(ecg_net, input_size=(1, 3600))

    #bottleneck_models = []
    for bottleneck_fraction in [0.75,0.5,0.25]:

        for conv_index in range(1,7): # we do not make the 1st layer a bottleneck one
            model = ECGNetWithBottleneckLayer(ecg_net,conv_index, bottleneck_fraction).cuda()
            # I think this is not actually working
            # load trained model
            model_path = model.name + "_trained_500_epochs.pth"
            model.load_state_dict(torch.load(model_path))
            _, test_acc = test(model, device, test_loader=entire_test_loader, epoch=499)
            print(f"{model.name} has accuracy {test_acc}")
            
    # for _model in bottleneck_models:
    #     _, test_acc = test(_model, device, test_loader=entire_test_loader, epoch=499)
    #     print(f"{_model.name} has accuracy {test_acc}")












    # TODO: test accuracy using the entire test set for all models
    # use entire_test_loader


    # # plot test accuracies
    # for key in [0,1,2]:
    #     # load dict with training/testing plots
    #     try:
    #         with open("mary_trained_networks.json", 'r') as json_file:
    #             dataGPU = json.load(json_file)
    #     except:
    #         dataGPU = {}
    #         print("Could not open file with trained models.")
    #     #print(f"The keys in dataGPU for key {key}: {dataGPU.keys()}")
    #     # List of keys to be removed
    #     keys_to_remove = []
    #     if key == 0:
    #         keys_to_remove.extend([_key for _key in dataGPU.keys() if _key.endswith('1')])
    #         keys_to_remove.extend([_key for _key in dataGPU.keys() if _key.endswith('2')])
    #     elif key == 1:
    #         keys_to_remove.extend([_key for _key in dataGPU.keys() if _key.endswith('0')])
    #         keys_to_remove.extend([_key for _key in dataGPU.keys() if _key.endswith('2')])
    #     elif key == 2:
    #         keys_to_remove.extend([_key for _key in dataGPU.keys() if _key.endswith('0')])
    #         keys_to_remove.extend([_key for _key in dataGPU.keys() if _key.endswith('1')])
    #     else:
    #         keys_to_remove = [] # should not happen
    #         print("You did not remove any keys.")
    #     #print(f"Keys to remove: {keys_to_remove}")
    #     for k in keys_to_remove:
    #         try:
    #             del dataGPU[k]
    #         except KeyError:
    #             # Handle the case where the key might not exist (though it should)
    #             print("KeyError")
    #     print(f"Keys in dataGPU after removing keys_to_remove: {dataGPU.keys()}")
    #     plot_test_acc_for_specific_frac(data=dataGPU, fraction=key)
    #     dataGPU = {}
    
    # # plotting all models
    # try:
    #     with open("mary_trained_networks.json", 'r') as json_file:
    #         dataGPU = json.load(json_file)
    # except:
    #     dataGPU = {}
    # #print(f"The keys in your json: {dataGPU.keys()}")
    # plot_all_models_test_acc(data=dataGPU)
    
    


if __name__ == '__main__':
    main()
    print("------all-done-----")
