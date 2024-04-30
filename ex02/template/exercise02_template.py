from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

class Linear():
    def __init__(self, in_features: int, out_features: int, batch_size: int, lr=0.1):
        super(Linear, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.weight = torch.randn(in_features, out_features) * np.sqrt(1. / in_features)
        self.bias = torch.randn(out_features) * np.sqrt(1. / in_features)
        self.grad_weight = torch.zeros(in_features, out_features)
        self.grad_bias = torch.zeros(out_features)
        self.input = torch.zeros(batch_size, in_features)

    def forward(self, input):
        self.input = input
        output = torch.mm(input, self.weight) + self.bias
        return output

    def backward(self, grad_output):
        grad_input = torch.mm(grad_output, self.weight.t())
        self.grad_weight = torch.mm(self.input.t(), grad_output)
        self.grad_bias = torch.sum(grad_output, dim=0)
        return grad_input

    def update(self):
        self.weight -= self.lr * self.grad_weight / self.batch_size
        self.bias -= self.lr * self.grad_bias / self.batch_size

class Sigmoid():
    def __init__(self, in_features: int, batch_size: int):
        super(Sigmoid, self).__init__()
        self.input = torch.zeros(batch_size)

    def forward(self, input):
        self.input = input
        output = 1 / (1 + torch.exp(-self.input))
        return output

    def backward(self, grad_output):
        grad_input = torch.sigmoid(self.input) * ((1 - torch.sigmoid(self.input))) * grad_output
        return grad_input

def Softmax(input):
    output = F.softmax(input, dim=1) #torch.exp(input - torch.max(input, dim=1, keepdim=True)[0]) / torch.sum(torch.exp(input - torch.max(input, dim=1, keepdim=True)[0]), dim=1, keepdim=True)
    return output

def compute_loss(target, prediction):
    return -torch.sum(target * torch.log(prediction))/prediction.shape[0]

def compute_gradient(target, prediction):
    return (prediction - target)

class MLP():
    def __init__(self, batch_size, lr):
        super(MLP, self).__init__()
        self.linear0 = Linear(28*28, 512, batch_size, lr)
        self.sigmoid0 = Sigmoid(512, batch_size)
        self.linear1 = Linear(512, 128, batch_size, lr)
        self.sigmoid1 = Sigmoid(128, batch_size)
        self.linear2 = Linear(128, 10, batch_size, lr)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear0.forward(x)
        x = self.sigmoid0.forward(x)
        x = self.linear1.forward(x)
        x = self.sigmoid1.forward(x)
        x = self.linear2.forward(x)
        x = Softmax(x)
        return x

    def backward(self, x):
        x = self.linear2.backward(x)
        x = self.sigmoid1.backward(x)
        x = self.linear1.backward(x)
        x = self.sigmoid0.backward(x)
        x = self.linear0.backward(x)

    def update(self):
        self.linear0.update()
        self.linear1.update()
        self.linear2.update()

def train(args, model, train_loader, epoch):
    train_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model.forward(data)
        loss = compute_loss(target, output)
        gradient = compute_gradient(target, output)
        model.backward(gradient)
        model.update()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / data.shape[0]))
    train_loss/= len(train_loader.dataset)
    return train_loss


def test(args, model, test_loader, epoch):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model.forward(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
       
        target = F.one_hot(target)
        loss = compute_loss(target, output)
        test_loss += loss

    test_loss /= len(test_loader.dataset)
    accuracy=correct / len(test_loader.dataset)
    print('\nTest Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss,accuracy

def loss_visualisation(train_loss,test_loss,epochs):
    plt.plot(range(epochs),train_loss,label='Train Loss',c='g')
    plt.plot(range(epochs),test_loss,label='Test Loss',c='r')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def accuracy_plot(accuracy,epochs,lr=0.1):
    plt.plot(range(epochs),accuracy,label="Learning Rate: {:.3f}".format(lr))
    plt.xlabel("Epochs")
    plt.ylabel("TestAccuracy")
    plt.legend()
    plt.show()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset_train = datasets.MNIST('../data', train=True, download=True,
                       transform=transform,
                       target_transform=torchvision.transforms.Compose([
                                 lambda x:torch.LongTensor([x]), # or just torch.tensor
                                 lambda x:F.one_hot(x, 10),
                                 lambda x:x.squeeze()]))

    dataset_test = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size = args.batch_size)
    test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size = args.batch_size)

    with torch.no_grad():
        #Train the network with the given default parameters for 30 epochs. Then plot how the test and
        #train loss develop with the number of epochs. Additionally create a plot, which shows how the test
        #accuracy develops.
        train_loss_history,test_loss_history,accuracy_history=[],[],[]
        model = MLP(args.batch_size, args.lr)
        for epoch in range(1, args.epochs + 1):
            train_loss=train(args, model, train_loader, epoch)
            train_loss_history.append(train_loss)
            test_loss,accuracy=test(args, model, test_loader, epoch)
            test_loss_history.append(test_loss)
            accuracy_history.append(accuracy)
        print(train_loss_history,test_loss_history,accuracy)
        loss_visualisation(train_loss_history,test_loss_history,epoch)
        accuracy_plot(accuracy_history,epoch)
        

        #Run the training again for 30 epochs, but this time with varying learning rates. Choose at least
        #five different learning rates between 1. and 0.001. Create a plot with the test accuracy over epochs
        #to compare the different learning rates.

        learning_rates = [1.0,0.5, 0.1, 0.01, 0.001] # choose different learning rates
        accuracy_history=[]
        for lr in learning_rates:
            model = MLP(args.batch_size, args.lr)
            optimizer = optim.SGD(model.parameters, lr=lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
            print("New LR:", scheduler.get_last_lr()[0])
            accuracy_list = []
            for epoch in range(1, args.epochs + 1):
                train_loss = train(args, model, train_loader, epoch)
                optimizer.step()
                test_loss, accuracy = test(args, model, test_loader, epoch)
                accuracy_list.append(accuracy)
                scheduler.step()
            accuracy_plot(accuracy_list,args.epochs,lr)


if __name__ == '__main__':
    main()
