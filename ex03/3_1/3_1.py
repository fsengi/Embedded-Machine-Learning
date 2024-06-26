from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import time
import json


class MLP(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.linear0 = nn.Linear(28*28, 512, bias=True)
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
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()    

    print(f'cuda is avail: {torch.cuda.is_available()} args.no_cuda: {args.no_cuda}')
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        print("WILL USE CUDA!!")
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    else:
        print("NO CUDA")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset_train = datasets.MNIST('../../data', train=True, download=True,
                       transform=transform)
    dataset_test = datasets.MNIST('../../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = MLP().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    timeDevice = []
    accDevice = []

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        accDevice.append(test(model, device, test_loader))
        timeDevice.append(time.time() - start)

    data = {
        'time': timeDevice,
        'acc': accDevice
    }

    if use_cuda:
        with open("GPUdata.json", 'w') as json_file:
            json.dump(data, json_file)
    else:
        with open("CPUdata.json", 'w') as json_file:
            json.dump(data, json_file)

if __name__ == '__main__':
    main()
