from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from torchvision import datasets, transforms
import time, json
import matplotlib.pyplot as plt
import numpy as np


class VGG11(nn.Module):
    def __init__(self):
        super().__init__()
        #self.layers = self._make_layers()
        self.name = "VGG11"
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.flat = nn.Flatten(end_dim=-1)
        self.lin1 = nn.Linear(in_features=512, out_features=4096)
        self.lin2 = nn.Linear(in_features=4096, out_features=4096)
        self.lin3 = nn.Linear(in_features=4096, out_features=10)

    def _make_layers(self):
        layers = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(end_dim=-1),

            nn.Linear(in_features=512, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=10)           
        ]
        return nn.ModuleList(layers)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), (2, 2))
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(F.relu(self.conv8(x)), (2, 2))
        #x = torch.flatten(x,end_dim=-1)
        x =self.flat(x)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return F.log_softmax(x, dim=1)



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % args.log_interval == 0:
            trainloss = 100. * batch_idx / len(train_loader)
            print('Current time: {:.4f}; Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.time(),
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

    print(f'Current time: {time.time()}; Test Epoch: {epoch}, Test set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc}%)\n')
    return test_loss, acc



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        print("Using CUDA")

    transform=transforms.Compose([transforms.ToTensor()])
    
    #cifar10_dataset_train = datasets.CIFAR10('../../data', train=True, download=True, transform=transform)
    #cifar10_dataset_test = datasets.CIFAR10('../../data', train=False, transform=transform)

    SVHN_dataset_train = datasets.SVHN('../../data', split = "train", download=True, transform=transform)
    SVHN_dataset_test = datasets.SVHN('../../data', split = "test", download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(SVHN_dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(SVHN_dataset_test, **test_kwargs)

    model = VGG11()

    for name, module in model.named_modules():
    #for name, module in model.layers:
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', n=1, dim=0, amount=0.1)
            print("Pruned Conv2D")
        # prune 40% of connections in all linear layers
        elif isinstance(module, nn.Linear):
            prune.ln_structured(module, name='weight', n=1, dim=0, amount=0.1)
            print("Pruned Linear")
    
    model = model.to(device)

    model_name = model.name
    print(f"Training model {model_name}")
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    timeDevice = []
    trainLoss = []
    testLoss = []
    accTest = []
    accTrain = []

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        trainloss, trainacc = train(args, model, device, train_loader, optimizer, epoch)
        testloss, testacc = test(model, device, test_loader, epoch)
        timeDevice.append(time.time() - start)
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
        
    with open("data.json", 'w') as json_file:
        json.dump(dataDict, json_file, indent=4)


if __name__ == '__main__':
    main()
