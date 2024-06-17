from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time, json
import torchvision.ops as tv_nn
from torchinfo import summary  # Import torchinfo for model summary


class VGG11(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.layers = self._make_layers(dropout_p)

    def _make_layers(self, dropout_p):
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
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=10)          
        ]
        return nn.ModuleList(layers)

    def forward(self, x):       
        for mod in self.layers:
            x = mod(x)
        output = F.log_softmax(x, dim=1)
        return output

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
    parser.add_argument('--dropout_p', type=float, default=0.0,
                        help='dropout_p (default: 0.0)')
    parser.add_argument('--L2_reg', type=float, default=None,
                        help='L2_reg (default: None)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--augment', type=int,  default=0, metavar='A',
                        help='use augmentation on test set (possible options =>1-3)')
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

        with open("GPUdata.json", 'r') as json_file:
            data = json.load(json_file)
    else:
        with open("CPUdata.json", 'r') as json_file:
            data = json.load(json_file)

    train_transform1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_transform2 = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_transform3 = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])



    transform=transforms.Compose([
        transforms.ToTensor()
        ])
    
    if (args.augment == 0):
        cifar10_dataset_train = datasets.CIFAR10('../../data', train=True, download=True, transform=transform)
    elif (args.augment == 1):
        cifar10_dataset_train = datasets.CIFAR10('../../data', train=True, download=True, transform=train_transform1)
    elif (args.augment == 2):
        cifar10_dataset_train = datasets.CIFAR10('../../data', train=True, download=True, transform=train_transform2)
    elif (args.augment == 3):
        cifar10_dataset_train = datasets.CIFAR10('../../data', train=True, download=True, transform=train_transform3)
    else:
        cifar10_dataset_train = datasets.CIFAR10('../../data', train=True, download=True, transform=transform)

    
    cifar10_dataset_test = datasets.CIFAR10('../../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(cifar10_dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(cifar10_dataset_test, **test_kwargs)

    model = VGG11(dropout_p=args.dropout_p).to(device)

    # Calculate MACs using torchinfo
    input_size = (1, 3, 32, 32)  # SVHN input size
    model_summary = summary(model, input_size=input_size, col_names=["input_size", "output_size", "num_params", "mult_adds"])
    total_macs = model_summary.total_mult_adds
    print(f"Total MACs: {total_macs}")

    if args.L2_reg is None:
        weight_decay = 0.0
    else:
        weight_decay = args.L2_reg
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=(weight_decay))
  
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

    if (args.L2_reg is not None):
        f_name = f'trained_VGG11_L2-{args.L2_reg}.pt'
        torch.save(model.state_dict(), f_name)
        print(f'Saved model to: {f_name}')

    dataDict = {
        'time': timeDevice,
        'trainLoss': trainLoss,
        'testLoss': testLoss,
        'accTest': accTest,
        'accTrain': accTrain
    }

    if (args.L2_reg != None):
        print('wdecay_')
        data[f'wdecay_{args.L2_reg}'] = dataDict
    elif (args.augment != 0.0):
        print('augment_')
        data[f'augment_{args.augment}'] = dataDict
    elif (args.dropout_p != 0.0):
        print('dropout_')
        data[f'dropout_{args.dropout_p}'] = dataDict
    else:
        print('baseline')
        data[f'baseline'] = dataDict

    if use_cuda:
        with open("GPUdata.json", 'w') as json_file:
            json.dump(data, json_file, indent=4)
    else:
        with open("CPUdata.json", 'w') as json_file:
            json.dump(data, json_file, indent=4)


if __name__ == '__main__':
    main()
