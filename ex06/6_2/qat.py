from __future__ import print_function
import argparse

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
# visualization/store imports
import time, json
import matplotlib.pyplot as plt
# brevitas imports
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint
from brevitas.quant.scaled_int import Int32Bias
from brevitas import config

config.IGNORE_MISSING_KEYS = True

class VGG11(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "VGG11"

        self.inp = nn.Identity()

        self.features = nn.Sequential(
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
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=10) 
        )
    
    def forward(self, x):
        x = self.inp(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

class QuantVGG11(nn.Module):
    """Quantized version of VGG11 with option to use different widths for activations and weights."""
    def __init__(self, weight_bit_width=4, act_bit_width=4):
        
        super().__init__()
        if weight_bit_width == act_bit_width:
            self.bit_width = weight_bit_width
            self.model_name = f"QUANTVGG11 with bitwidth={self.bit_width}"
        else:
            self.act_bitwidth = act_bit_width
            self.weight_bitwidth = weight_bit_width
            self.model_name = f"QUANTVGG11 with activation bitwidth={act_bit_width} and weight bitwidth {weight_bit_width}."
        
        self.quant_inp = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)
        
        self.features = nn.Sequential(
            qnn.QuantConv2d(3, 64, kernel_size=3, padding=1, bias=True, weight_bit_width=weight_bit_width, bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            qnn.QuantConv2d(64, 128, kernel_size=3, padding=1, bias=True, weight_bit_width=weight_bit_width, bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            qnn.QuantConv2d(128, 256, kernel_size=3, padding=1, bias=True, weight_bit_width=weight_bit_width, bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),
            
            qnn.QuantConv2d(256, 256, kernel_size=3, padding=1, bias=True, weight_bit_width=weight_bit_width, bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            qnn.QuantConv2d(256, 512, kernel_size=3, padding=1, bias=True, weight_bit_width=weight_bit_width, bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),
            
            qnn.QuantConv2d(512, 512, kernel_size=3, padding=1, bias=True, weight_bit_width=weight_bit_width, bias_quant=Int32Bias),
            qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            qnn.QuantConv2d(512, 512, kernel_size=3, padding=1, bias=True, weight_bit_width=weight_bit_width, bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),
            
            qnn.QuantConv2d(512, 512, kernel_size=3, padding=1, bias=True, weight_bit_width=weight_bit_width, bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            qnn.QuantLinear(512, 4096, bias=True, weight_bit_width=weight_bit_width, bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),
            qnn.QuantLinear(4096, 4096, bias=True, weight_bit_width=weight_bit_width, bias_quant=Int32Bias, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),
            qnn.QuantLinear(4096, 10, bias=True, weight_bit_width=weight_bit_width, bias_quant=Int32Bias, return_quant_tensor=True),        
        )
    
    def forward(self, x):
        x = self.quant_inp(x) # quantize input 
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x


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

def plot_one_model(data:dict, epochs, title) -> None:
    # Create the main plot
    fig, ax1 = plt.subplots()

    # Plot the first set of data
    ax1.plot(data['accTest'], 'bo-', label='accuracy')
    ax1.set_xlabel("time in s")
    ax1.set_ylabel("accuracy in %", color='b')
    ax1.tick_params('y', colors='b')

    # Create a secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(data['trainLoss'], 'x-', label='trainLoss')
    ax2.plot(data['testLoss'], 'x-', label='testLoss')
    ax2.set_ylabel('Loss')
    ax2.tick_params('y')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')


    # Show the plot
    plt.title(title)
    plt.legend()
    plt.savefig(title + ".png")


def transfer_weights(pretrained_model, quantized_model):
    pretrained_dict = pretrained_model.state_dict()
    quantized_model_dict = quantized_model.state_dict()
    
    # only keep matching keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in quantized_model_dict and 'weight' in k or 'bias' in k}
    
    quantized_model_dict.update(pretrained_dict)
    return quantized_model


def calibrate_model(model, loader, device):
    model.eval()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            model(data)

    return model

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
    parser.add_argument('--bit-widths', type=int, default=4, metavar='N', nargs='+',
                        help='how many bits for quantization')
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

    transform=transforms.Compose([transforms.ToTensor()])
    
    cifar10_dataset_train = datasets.CIFAR10('../../data', train=True, download=True, transform=transform)
    cifar10_dataset_test = datasets.CIFAR10('../../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(cifar10_dataset_train, **train_kwargs) 
    test_loader = torch.utils.data.DataLoader(cifar10_dataset_test, **test_kwargs)


    bit_widths = args.bit_widths
    accurracies = []

    for bit_width in bit_widths:
        print("------------------------------------")
        print(f"bit width is {bit_width}")
        
        # initialize
        model = QuantVGG11(bit_width, bit_width)

        model = model.to(device)

        model_name = model.model_name
        print(f"Training model {model_name} for {args.epochs} epochs")
        
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

        accurracies.append(testloss)

        dataDict = {
            'time': timeDevice,
            'trainLoss': trainLoss,
            'testLoss': testLoss,
            'accTest': accTest,
            'accTrain': accTrain
        }

        data[model_name] = dataDict

        
        with open("GPUdata.json", 'w') as json_file:
            json.dump(data, json_file, indent=4)

        plot_one_model(dataDict,epochs=args.epochs, title=f"{model_name}_trained_{args.epochs}_epochs")

        model_path = f"{model.model_name}_trained_{args.epochs}_epochs.pth"
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')
    
    plt.figure()
    plt.plot(bit_widths, accurracies)
    plt.xlabel("Bitwidth")
    plt.ylabel("Test accuracy (%)")
    plt.title("Quantization_aware_trained_network")
    plt.legend()
    plt.savefig("Quantization_aware_trained_network" + ".png")

if __name__ == '__main__':
    main()
