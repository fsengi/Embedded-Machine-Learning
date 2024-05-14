from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time

# TODO: Implement the MLP class, to be equivalent to the MLP from the last exercise!
class MLP(nn.Module):
<<<<<<<< Updated upstream:ex03/3_2/exercise03_template.py
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(32*32*3, 512)
========
    def __init__(self) -> None:
        super().__init__()
        self.linear0 = nn.Linear(32*32*3, 512, bias=True)
>>>>>>>> Stashed changes:ex03/3_2/3_2.py
        self.sigmoid0 = nn.Sigmoid()
        self.linear1 = nn.Linear(512, 128)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
      x = torch.flatten(x, 1)
      x = self.linear0(x)
      x = self.sigmoid0(x)
      x = self.linear1(x)
      x = self.sigmoid1(x)
      x = self.linear2(x)
      x = F.log_softmax(x, dim=1)
      return x

<<<<<<<< Updated upstream:ex03/3_2/exercise03_template.py

# TODO: Implement the CNN class, as defined in the exercise!
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.linear0 = nn.Linear(128 * 12 * 12, 128)
        self.relu3 = nn.ReLU()
        self.linear1 = nn.Linear(128, 10)

    def forward(self, x):
      x = self.conv0(x)
      x = self.relu0(x)
      x = self.conv1(x)
      x = self.relu1(x)
      x = self.conv2(x)
      x = self.relu2(x)
      x = torch.flatten(x, 1)
      x = self.linear0(x)
      x = self.relu3(x)
      x = self.linear1(x)
      x = F.log_softmax(x, dim=1)
      return x


========
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
>>>>>>>> Stashed changes:ex03/3_2/3_2.py

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
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
<<<<<<<< Updated upstream:ex03/3_2/exercise03_template.py
                        help='learning rate (default: 0.1)')
========
                        help='learning rate (default: 1.0)')
>>>>>>>> Stashed changes:ex03/3_2/3_2.py
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
<<<<<<<< Updated upstream:ex03/3_2/exercise03_template.py
    print('Device: ' + ("GPU" if use_cuda else "CPU"))
========
>>>>>>>> Stashed changes:ex03/3_2/3_2.py

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    #load MNIST dataset as dataset0
    transform=transforms.Compose([
        transforms.ToTensor()
        ])
<<<<<<<< Updated upstream:ex03/3_2/exercise03_template.py
    dataset_train0 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset_test0 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader0 = torch.utils.data.DataLoader(dataset_train0,**train_kwargs)
    test_loader0 = torch.utils.data.DataLoader(dataset_test0, **test_kwargs)

    #load CIFAR dataset as dataset1
    dataset_train1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.ToTensor())
    dataset_test1 = datasets.CIFAR10('../data', train=False,
                       transform=transforms.ToTensor())
    train_loader1 = torch.utils.data.DataLoader(dataset_train1,**train_kwargs)
    test_loader1 = torch.utils.data.DataLoader(dataset_test1, **test_kwargs)


    model0 = MLP().to(device)

    optimizer = optim.SGD(model0.parameters(), lr=args.lr)

    times = []
    accuracys = []

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model0, device, train_loader1, optimizer, epoch)
        accuracys.append(test(model0, device, test_loader1))
        times.append(time.time() - start)

    data = {
        'time': times,
        'accuracy': accuracys
    }
    with open("MLP.json", 'w') as json_file:
            json.dump(data, json_file)

"""
    model1 = CNN().to(device)

    optimizer = optim.SGD(model1.parameters(), lr=args.lr)

    times = []
    accuracys = []

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model1, device, train_loader1, optimizer, epoch)
        accuracys.append(test(model1, device, test_loader1))
        times.append(time.time() - start)

    data = {
        'time': times,
        'accuracy': accuracys
    }
    with open("CNN.json", 'w') as json_file:
            json.dump(data, json_file)
"""

========
    
    cifar10_dataset_train = datasets.CIFAR10('../../data', train=True, download=True, transform=transform)
    cifar10_dataset_test = datasets.CIFAR10('../../data', train=False, transform=transform)

    cifar10_train_loader = torch.utils.data.DataLoader(cifar10_dataset_train,**train_kwargs)
    cifar10_test_loader = torch.utils.data.DataLoader(cifar10_dataset_test,**test_kwargs)

    modelList = []
    modelList.append( MLP().to(device) )
    modelList.append( CNN().to(device) )

    accMLP = []
    accCNN = []
    masterListAcc= [accMLP, accCNN]

    tMLP = []
    tCNN = []
    masterListT= [tMLP, tCNN]

    for model, accList, tList in zip(modelList, masterListAcc, masterListT):
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, cifar10_train_loader, optimizer, epoch)
            accList.append(test(model, device, cifar10_test_loader))
            tList.append(time.time() - start)
    
    plotdataAcc(mlp=accMLP, cnn=accCNN)
    plotdataTime(mlptime=tMLP, mlpacc=accMLP, cnnacc=accCNN, cnntime=tCNN)

def plotdataAcc(mlp:list, cnn:list) -> None:
    plt.clf()
    plt.plot(mlp, label='MLP')
    plt.plot(cnn, label='CNN')
    plt.title("accuracy over epochs for MLP and CNN")
    plt.ylabel("accuracy in %")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("acc_plot_3_2.png")

def plotdataTime(mlpacc:list, cnnacc:list, mlptime:list, cnntime:list) -> None:
    plt.clf()
    plt.plot(mlptime, mlpacc, label='MLP')
    plt.plot(cnntime, cnnacc, label='CNN')
    plt.title("accuracy over executiontime for MLP and CNN on GPU")
    plt.ylabel("accuracy in %")
    plt.xlabel("time in s")
    plt.legend()
    plt.savefig("timeplot_3_2.png")
>>>>>>>> Stashed changes:ex03/3_2/3_2.py

if __name__ == '__main__':
    main()
