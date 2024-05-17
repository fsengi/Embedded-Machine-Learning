import json
import matplotlib.pyplot as plt
import argparse
import numpy as np


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    
    parser.add_argument('--plot-bar', action='store_true', default=False,
                        help='plot time diff cpu gpu')
    parser.add_argument('--plot-loss', action='store_true', default=False,
                        help='plot loss no dropout')
    parser.add_argument('--plot-dropout', action='store_true', default=False,
                        help='plot loss with dropout')
  
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

