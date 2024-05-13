from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import time
import json


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear0 = nn.Linear(32*32*3, 512, bias=True)
        self.sigmoid0 = nn.Sigmoid()
        self.linear1 = nn.Linear(512, 128, bias=True)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(128, 10, bias=True)

        self.parameters = nn.ParameterList([self.linear0.weight, self.linear0.bias, self.linear1.weight, self.linear1.bias,self.linear2.weight, self.linear2.bias])

    def forward(self, x):
      x = torch.flatten(x, 1)
      x = self.linear0(x)
      x = self.sigmoid0(x)
      x = self.linear1(x)
      x = self.sigmoid1(x)
      x = self.linear2(x)
      x = F.log_softmax(x, dim=1)
      return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Layer 1: Convolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        # Layer 2: Convolution
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        # Layer 3: Convolution
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        # Layer 5: Linear
        self.fc1 = nn.Linear(18432, 128)  # Calculating input features based on image size
        # Layer 6: Linear
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Layer 1: Convolution -> ReLU
        x = F.relu(self.conv1(x))
        # Layer 2: Convolution -> ReLU
        x = F.relu(self.conv2(x))
        # Layer 3: Convolution -> ReLU
        x = F.relu(self.conv3(x))
        # Flatten layer
        x = torch.flatten(x, 1)
        # x = x.view(-1, 128 * 11 * 11)  # Reshape for fully connected layer
        # Layer 5: Linear -> ReLU
        x = F.relu(self.fc1(x))
        # Layer 6: Linear -> LogSoftmax
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(model, device, test_loader):
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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor()
        ])
    
    cifar10_dataset_train = datasets.CIFAR10('../../data', train=True, download=True, transform=transform)
    cifar10_dataset_test = datasets.CIFAR10('../../data', train=False, transform=transform)

    cifar10_train_loader = torch.utils.data.DataLoader(cifar10_dataset_train,**train_kwargs)
    cifar10_test_loader = torch.utils.data.DataLoader(cifar10_dataset_test,**test_kwargs)

    # optimizersList = ['SGD', 'Adam','Adagrad','Adadelta','Rprop','RAdam']
    optimizersList = ['Adam','Adagrad','Adadelta','Rprop','RAdam']

    lrList = [0.1, 0.05, 0.01, 0.005, 0.001]

    with open("data.json", 'r') as json_file:
        datadict = json.load(json_file)

    for optimizerStr in optimizersList:
        datadict[optimizerStr] = {}
        for lr in lrList:
            print(f'optimizerStr {optimizerStr} lr {lr}')
            model = CNN().to(device)
            accList = []
            if optimizerStr == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=lr)
            elif optimizerStr == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
            elif optimizerStr == 'Adagrad':
                optimizer = optim.Adagrad(model.parameters(), lr=lr)
                # optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, foreach=None, *, maximize=False, differentiable=False)
            elif optimizerStr == 'Adadelta':
                optimizer = optim.Adadelta(model.parameters(), lr=lr)
            elif optimizerStr == 'Rprop':
                optimizer = optim.Rprop(model.parameters(), lr=lr)
            elif optimizerStr == 'RAdam':
                optimizer = optim.RAdam(model.parameters(), lr=lr)

            for epoch in range(1, args.epochs + 1):
                train(args, model, device, cifar10_train_loader, optimizer, epoch)
                accList.append(test(model, device, cifar10_test_loader))

            datadict[optimizerStr][str(lr)] = accList
            plotdataAcc(acc=accList, name=optimizerStr, lr=lr)
        plt.clf()

        with open("data.json", 'w') as json_file:
            json.dump(datadict, json_file)     


def plotdataAcc(acc:list, name:str, lr:float) -> None:
    plt.plot(acc, label=lr)
    plt.title(f"accuracy over epochs for {name} CNN optimizers")
    plt.ylabel("accuracy in %")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig(f"acc_plot_3_3_{name}.png")

if __name__ == '__main__':
    main()

