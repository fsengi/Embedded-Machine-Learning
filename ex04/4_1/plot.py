from ex4_1 import VGG11
import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch


def plotdata(data:dict) -> None:
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
    plt.title("VGG11 accuracy, test loss and train loss over 30 epochs")
    plt.legend()
    plt.savefig("4_1_loss_ocer_time.png")

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
    left_value = dataCPU['time'][-1]
    right_value = dataGPU['time'][-1]

    # Position of the bars on the x-axis
    x_positions = [0, 1]  # Two positions: 0 for the left bar and 1 for the right bar

    # Create the plot
    fig, ax = plt.subplots()

    # Plot the left side bar
    barcpu = ax.bar(x_positions[0], left_value, width=0.4, label=f'{left_label} {round(left_value,2)}s')

    # Plot the right side bar
    bargpu = ax.bar(x_positions[1], right_value, width=0.4, label=f'{right_label} {round(right_value,2)}s')

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
    ax1.plot(data["dropout_dropout_0.0"]['accTest'], 'bo-', label='accuracy dropout 0.0')
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


def plotWeightDecayAcc(data:dict) -> None:
    # Plot the first set of data
    plt.plot(data["wdecay_0.0"]['accTest'], 'x-', label='L2 reg 0.0')
    plt.plot(data["wdecay_0.001"]['accTest'], 'x-', label='L2 reg 0.001')
    plt.plot(data["wdecay_0.0001"]['accTest'], 'x-', label='L2 reg 0.0001')
    plt.plot(data["wdecay_1e-05"]['accTest'], 'x-', label='L2 reg 0.00001')
    plt.plot(data["wdecay_1e-06"]['accTest'], 'x-', label='L2 reg 0.000001')
    plt.xlabel("epochs")
    plt.ylabel("accuracy in %")
    # Add legends
    plt.legend()
    # Show the plot
    plt.title("VGG11 accuracy for different L2 regularization rates ")
    plt.savefig("4_2_L2_regularization_acc.png")
    plt.ylim(65, 75)
    plt.savefig("4_2_L2_regularization_acc_zoonm.png")


def plotWeightDecayHist(weight_list:list, titels:list) -> None:
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    for i, (weights, title) in enumerate(zip(weight_list, titles)):
        axs[i].hist(weights.flatten(), bins=30, color='blue', alpha=0.7)
        axs[i].set_title(title)
        axs[i].set_xlabel('epochs')
        axs[i].set_ylabel('weight value')
    plt.tight_layout()
    # Show the plot
    plt.title("VGG11 last layer histograms for different L2 regularization rates over epochs")
    plt.legend()
    plt.savefig("4_2_L2_reg_hist.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    
    parser.add_argument('--plot-bar', action='store_true', default=False,
                        help='plot time diff cpu gpu')
    parser.add_argument('--plot-loss', action='store_true', default=False,
                        help='plot loss no dropout')
    parser.add_argument('--plot-dropout', action='store_true', default=False,
                        help='plot loss with dropout')
    parser.add_argument('--plot-weight-decay', action='store_true', default=False,
                        help='plot hist for weight decay')        
  
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
    elif args.plot_loss:
        plotdata(data=dataGPU)
    elif args.plot_dropout:
        plotdropout(data=dataGPU)
    elif args.plot_weight_decay:
        plotWeightDecayAcc(data=dataGPU)
        # plotWeightDecayHist(data=dataGPU)

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
        for filepath in model_files:
            model.state_dict(torch.load(filepath), 'weights')
            
            for param_tensor in model.state_dict():
                print(param_tensor, "\t", model.state_dict()[param_tensor].size())

