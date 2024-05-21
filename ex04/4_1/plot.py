import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import torch.nn as nn

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

def plotdata(data:dict) -> None:
    # Create the main plot
    fig, ax1 = plt.subplots()

    # Plot the first set of data
    ax1.plot(data['baseline']['accTest'], 'bo-', label='accuracy')
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy in %", color='b')
    ax1.tick_params('y', colors='b')

    # Create a secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(data['baseline']['trainLoss'], 'x-', label='trainLoss')
    ax2.plot(data['baseline']['testLoss'], 'x-', label='testLoss')
    ax2.set_ylabel('Loss')
    ax2.tick_params('y')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.title("VGG11 accuracy, test loss and train loss over 30 epochs")
    plt.legend()
    plt.savefig("4_1_baseline_over_epochs.png")

def plotOverTime(data:dict) -> None:
    # Create the main plot
    fig, ax1 = plt.subplots()

    # Plot the first set of data
    ax1.plot(data['time'], data['accTest'], 'bo-', label='accuracy')
    ax1.set_xlabel("time in s")
    ax1.set_ylabel("accuracy in %", color='b')
    ax1.tick_params('y', colors='b')

    # Create a secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(data['time'], data['trainLoss'], 'x-', label='trainLoss')
    ax2.plot(data['time'], data['testLoss'], 'x-', label='testLoss')
    ax2.set_ylabel('Loss')
    ax2.tick_params('y')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.title("accuracy, test loss and train loss over time (30 epochs) on VGG11")
    plt.legend()
    plt.savefig("4_1_acc_loss_over_epochs.png")

def plotbar(dataCPU:dict, dataGPU:dict, ) -> None:
    # Example data
    left_label = 'CPU'
    right_label = 'GPU'
    left_value = dataCPU['baseline']['time'][0]
    right_value = dataGPU['baseline']['time'][0]

    # Position of the bars on the x-axis
    x_positions = [0, 1]  # Two positions: 0 for the left bar and 1 for the right bar

    # Create the plot
    fig, ax = plt.subplots()

    # Plot the left side bar
    ax.bar(x_positions[0], left_value, width=0.4, label=f'{left_label} {round(left_value,2)}s')

    # Plot the right side bar
    ax.bar(x_positions[1], right_value, width=0.4, label=f'{right_label} {round(right_value,2)}s')

    # Customize the plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels([left_label, right_label])
    ax.set_ylabel('time in s')
    ax.set_title('time comparison CPU vs. GPU over 1 epochs')
    ax.legend()

    # Show the plot
    plt.show()
    plt.savefig("4_1_time_comp.png")

def plotdropout(data:dict) -> None:
    # Create the main plot
    fig, ax1 = plt.subplots()

    # Plot the first set of data
    ax1.plot(data["augment_0"]['accTest'], 'bo-', label='accuracy without augmentation')
    # ax1.plot(data["dropout_0.1"]['accTest'], 'bo-', label='accuracy dropout 0.1')
    # ax1.plot(data["dropout_0.3"]['accTest'], 'bo-', label='accuracy dropout 0.3')
    # ax1.plot(data["dropout_0.5"]['accTest'], 'bo-', label='accuracy dropout 0.5')
    # ax1.plot(data["dropout_0.7"]['accTest'], 'bo-', label='accuracy dropout 0.7')
    ax1.plot(data["dropout_0.9"]['accTest'], 'ro-', label='accuracy dropout 0.9')

    ax1.set_xlabel("epochos")
    ax1.set_ylabel("accuracy in %", color='b')
    ax1.tick_params('y', colors='b')

    # Create a secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(np.array(data["dropout_0.1"]['accTest']) - np.array(data["dropout_0.0"]['accTest']), 'x-', label='diff 0.1')
    # ax2.plot(np.array(data["dropout_0.3"]['accTest']) - np.array(data["dropout_0.0"]['accTest']), 'x-', label='diff 0.3')
    ax2.plot(np.array(data["dropout_0.5"]['accTest']) - np.array(data["dropout_0.0"]['accTest']), 'x-', label='diff 0.5')
    # ax2.plot(np.array(data["dropout_0.7"]['accTest']) - np.array(data["dropout_0.0"]['accTest']), 'x-', label='diff 0.7')
    ax2.plot(np.array(data["dropout_0.9"]['accTest']) - np.array(data["dropout_0.0"]['accTest']), 'x-', label='diff 0.9')
    ax2.set_ylabel('diff Loss')
    ax2.tick_params('y')
    ax2.set_ylim(-4, 5)
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.title("VGG11 accuracy and accuracy difference for dropout rates ")
    plt.legend()
    plt.savefig("4_1_diff_dropout.png")

def plotaugment(data:dict) -> None:
    # Create the main plot
    fig, ax1 = plt.subplots()

    # Plot the first set of data
    ax1.plot(data["augment_0"]['accTrain'], 'b--', label='training accuracy without augmentation')
    ax1.plot(data["augment_1"]['accTrain'], 'mo-', label='training accuracy with randomcrop')
    ax1.plot(data["augment_2"]['accTrain'], 'go-', label='training accuracy with normalize')
    ax1.plot(data["augment_3"]['accTrain'], 'ro-', label='training accuracy with randomhorizontalflip')

    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy in %", color='b')
    ax1.tick_params('y', colors='b')

    # Show the plot
    plt.title("VGG11 accuracy with different data augmentations")
    plt.legend()
    plt.savefig("4_3_augmentation.png")

def plotaugmenttime(data:dict, ) -> None:
    # Example data
    baseline_label = 'baseline'
    firstaugment_label = 'randomcrop'
    secondaugment_label = 'normalize'
    thirdaugment_label = 'randomhorizontalflip'
    baseline_value = data["augment_0"]['time'][-1]
    firstaugment_value = data["augment_1"]['time'][-1]
    secondaugment_value = data["augment_2"]['time'][-1]
    thirdaugment_value = data["augment_3"]['time'][-1]

    # Position of the bars on the x-axis
    x_positions = [0, 1, 2, 3]  # Two positions: 0 for the left bar and 1 for the right bar

    # Create the plot
    fig, ax = plt.subplots()

    # Plot the left side bar
    ax.bar(x_positions[0], baseline_value, width=0.4, label=f'{baseline_label} {round(baseline_value,2)}s')

    ax.bar(x_positions[1], firstaugment_value, width=0.4, label=f'{firstaugment_label} {round(firstaugment_value,2)}s')

    ax.bar(x_positions[2], secondaugment_value, width=0.4, label=f'{secondaugment_label} {round(secondaugment_value,2)}s')

    ax.bar(x_positions[3], thirdaugment_value, width=0.4, label=f'{thirdaugment_label} {round(thirdaugment_value,2)}s')

    # Customize the plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels([baseline_label, firstaugment_label, secondaugment_label, thirdaugment_label])
    ax.set_ylabel('time in s')
    ax.set_title('time comparison baseline vs different dataset augmentation')
    ax.legend()

    # Show the plot
    plt.show()
    plt.savefig("4_3_augmentation_time_comp.png")

def plotaugmenttest(data:dict) -> None:
    # Create the main plot
    fig, ax1 = plt.subplots()

    # Plot the first set of data
    ax1.plot(data["augment_0"]['accTest'], 'b--', label='accuracy without augmentation')
    ax1.plot(data["augment_1"]['accTest'], 'mo-', label='accuracy with randomcrop')
    ax1.plot(data["augment_2"]['accTest'], 'go-', label='accuracy with normalize')
    ax1.plot(data["augment_3"]['accTest'], 'ro-', label='accuracy with randomhorizontalflip')
    ax1.set_xlabel("time in s")
    ax1.set_ylabel("accuracy in %", color='b')
    ax1.tick_params('y', colors='b')

    # Add legends
    ax1.legend(loc='upper left')

    # Show the plot
    plt.title("VGG11 test accuracy over 30 epochs with augmentation")
    plt.legend()
    plt.savefig("4_3_test_accuracy_loss.png")

def plotWeightDecayAcc(data:dict) -> None:
    # Plot the first set of data
    plt.plot(data["wdecay_0.0"]['accTest'], 'x-', label='L2 reg 0.0')
    plt.plot(data["wdecay_0.001"]['accTest'], 'x-', label='L2 reg 1e-3')
    plt.plot(data["wdecay_0.0001"]['accTest'], 'x-', label='L2 reg 1e-4')
    plt.plot(data["wdecay_1e-05"]['accTest'], 'x-', label='L2 reg 1e-5')
    plt.plot(data["wdecay_1e-06"]['accTest'], 'x-', label='L2 reg 1e-6')
    plt.xlabel("epochs")
    plt.ylabel("accuracy in %")
    # Add legends
    plt.legend()
    # Show the plot
    plt.title("VGG11 accuracy for different L2 regularization rates")
    plt.savefig("4_2_L2_regularization_acc.png")
    plt.ylim(65, 75)
    plt.savefig("4_2_L2_regularization_acc_zoom.png")


def plotWeightDecayHist(weight_list:list, titels:list) -> None:
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    for i, (weights, title) in enumerate(zip(weight_list, titels)):
        axs[i].hist(weights, bins=30, alpha=0.7)
        axs[i].set_title(title)
        axs[i].set_xlabel('epochs')
        axs[i].set_ylabel('count weight value')
    plt.tight_layout()
    # Show the plot
    # plt.title("VGG11 last layer histograms for different L2 regularization rates over epochs")
    plt.legend()
    plt.savefig("4_2_L2_reg_hist.png")

def plotSingleWeightDecayHist(weight_list:list, titel:str) -> None:
    plt.clf()
    plt.hist(weight_list[-1], bins=30, alpha=0.7)
    plt.title(titel)
    plt.xlabel('epochs')
    plt.ylabel('count weight value')
    plt.tight_layout()
    # Show the plot
    plt.title("VGG11 last layer histograms for different L2 regularization rates over epochs")
    plt.legend()
    plt.savefig(f"4_2_L2_reg_hist{titel}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    
    parser.add_argument('--plot-bar', action='store_true', default=False,
                        help='plot time diff cpu gpu')
    parser.add_argument('--plot-baseline', action='store_true', default=False,
                        help='plot baseline data no regularization no transform')
    parser.add_argument('--plot-dropout', action='store_true', default=False,
                        help='plot loss with dropout')
    parser.add_argument('--plot-weight-decay', action='store_true', default=False,
                        help='plot hist for weight decay')        
    parser.add_argument('--plot-augment', action='store_true', default=False,
                        help='plot accuracy and time for augmented dataset')   

    args = parser.parse_args()
    use_cuda = not args.plot_bar

    try:
        with open("GPUdata.json", 'r') as json_file:
            dataGPU = json.load( json_file)
    except:
        pass

    try:
        with open("CPUdata.json", 'r') as json_file:
            dataCPU = json.load( json_file)
    except:
        pass
    
    if args.plot_bar:
        plotbar(dataGPU=dataGPU, dataCPU=dataCPU)
    elif args.plot_baseline:
        plotdata(data=dataGPU)
    elif args.plot_dropout:
        plotdropout(data=dataGPU)
    elif args.plot_augment:
        plotaugment(data=dataGPU)
        plotaugmenttest(data=dataGPU)
        plotaugmenttime(data=dataGPU)
    elif args.plot_weight_decay:
        plotWeightDecayAcc(data=dataGPU)

        # File paths to your .pt files
        model_files = [
            'trained_VGG11_L2-0.0.pt',
            'trained_VGG11_L2-0.001.pt',
            'trained_VGG11_L2-0.0001.pt',
            'trained_VGG11_L2-1e-05.pt',
            'trained_VGG11_L2-1e-06.pt',
        ]

        # Titles for the subplots
        titles = ['L2 reg 0.0', 'L2 reg 0.001', 'L2 reg 1e-04', 'L2 reg 1e-05', 'L2 reg 1e-06']

        # Load models and extract last layer weights
        model = VGG11()

        weight_list = []

        for filepath, title in zip(model_files, titles):
            model = VGG11()
            model.state_dict(torch.load(filepath))
            
            for name, param in model.named_parameters():
                if 'layers.18.weight' in name:
                    weight_list.append([param.detach().view(-1).numpy()])
                    print(f'{weight_list[-1]} min{np.amin(weight_list[-1])} max{np.amax(weight_list[-1])}')
                    plotSingleWeightDecayHist(weight_list=weight_list, titel=title)
        plotWeightDecayHist(weight_list=weight_list, titels=titles)
    

