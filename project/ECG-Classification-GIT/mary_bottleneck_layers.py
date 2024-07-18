import argparse
import os # load data, models,...
import time # time training
from datetime import datetime
import json # log for training

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
        print(f'in_channels: {in_channels}, bottleneck_channels: {bottleneck_channels}, out_channels: {out_channels}')
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
        print(f"defined {self.name}")
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
    return train_loader, test_loader

def train_one_epoch(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % args.log_interval == 0:
            trainloss = 100. * batch_idx / len(train_loader)
            print('Current time: {}; Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                epoch, batch_idx * len(data), len(train_loader.dataset),
                trainloss,  loss.item()))
    return loss.item(), 100. * correct / total
    

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

    print(f"Current time: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}; Test Epoch: {epoch}, Test set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc}%)\n")
    return test_loss, acc


def plot_one_model(data:dict, epochs, title) -> None:
    # Create the main plot
    fig, ax1 = plt.subplots()

    # Plot the first set of data
    ax1.plot(data['accTest'], 'b-', label='accuracy')
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy in %", color='b')
    ax1.tick_params('y', colors='b')

    # Create a secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(data['trainLoss'], '-', label='trainLoss')
    ax2.plot(data['testLoss'], '-', label='testLoss')
    ax2.set_ylabel('Loss')
    ax2.tick_params('y')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')


    # Show the plot
    plt.title(title)
    plt.legend()
    plt.savefig(title + ".png")

def plot_test_acc_for_several_models(data:dict) -> None:
    for key in data.keys():
        _data = data[key]
        plt.plot(_data['accTest'], 'o-', label=str(key))


    # Show the plot
    plt.title("Test accuracy over 30 epochs")
    plt.legend()
    plt.show()
    plt.savefig("all_models.png")
    

def train(model, train_loader, test_loader, epochs, optimizer, criterion, args):
    print(f"Training model {model.name} for {epochs} epochs")

    with open(args.train_log_path, "r") as json_file:
        train_log = json.load(json_file)
    
    timeDevice = []
    trainLoss = []
    testLoss = []
    accTest = []
    accTrain = []

    start = time.time()
    for epoch in range(1, epochs + 1):
        trainloss, trainacc = train_one_epoch(args, model, args.device, train_loader, optimizer, criterion, epoch)
        testloss, testacc = test(model, args.device, test_loader, epoch)
        timeDevice.append((time.time() - start)/60) # difference in minutes
        accTest.append(testacc)
        accTrain.append(trainacc)
        testLoss.append(testloss)
        trainLoss.append(trainloss)


    dataDict = {
        'time': timeDevice,
        'trainLoss': trainLoss,
        'testLoss': testLoss,
        'accTest': accTest,
        'accTrain': accTrain
    }
    train_log[model.name] = dataDict

    
    with open(args.train_log_path, 'w') as json_file:
        json.dump(train_log, json_file, indent=4)

    plot_one_model(dataDict,epochs=args.epochs, title=f"{model.name}_trained_{args.epochs}_epochs")

    model_path = f"{model.name}_trained_{args.epochs}_epochs.pth"
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

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

    train_loader, test_loader = prepare_data(batch_size=args.batch_size)

    models = []

    # load classical model
    ecg_net = ECGNet().cuda()

    # train classical model
    train(
        model=ecg_net, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        epochs=args.epochs,
        optimizer=optim.Adam(ecg_net.parameters(), lr=args.lr),
        criterion=torch.nn.CrossEntropyLoss(),
        args=args,
    )


    summary(ecg_net, input_size=(1, 3600))

    # for bottleneck_fraction in [0.75,0.5,0.25]:

    #     bottleneck_models = []
        
    #     for conv_index in range(1,7): # we do not make the 1st layer a bottleneck one
    #         model = ECGNetWithBottleneckLayer(ecg_net,conv_index, bottleneck_fraction).cuda()

    #         bottleneck_models.append(model)

    
        # # TODO: train bottleneck models
        # for conv_index in range(1,7):
        #     print(f"Training {conv_index}th bottleneck model bottleneck fraction = {bottleneck_fraction}:")
        #     train(
        #         model=bottleneck_models[conv_index-1], 
        #         train_loader=train_loader, 
        #         test_loader=test_loader, 
        #         epochs=args.epochs,
        #         optimizer=optim.Adam(ecg_net.parameters(), lr=args.lr),
        #         criterion=torch.nn.CrossEntropyLoss(),
        #         args=args,
        #         )
        #     summary(bottleneck_models[conv_index-1], input_size=(1,3600))
    # TODO: average test accuracy over the last n epochs
    # TODO: test accuracy using the entire test set
    # TODO: plot all accuracies in one graph
    


if __name__ == '__main__':
    main()
    print("------all-done-----")
