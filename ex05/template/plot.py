import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import torch.nn as nn

def plotdata(data:dict) -> None:
    # Create the main plot
    fig, ax1 = plt.subplots()

    # Plot the first set of data
    ax1.plot(data['accTest'], 'bo-', label='test accuracy')
    ax1.plot(data['accTrain'], 'go-', label='training accuracy')
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy in %", color='b')
    ax1.tick_params('y', colors='b')

    # Create a secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(data['trainLoss'], 'x-', label='trainLoss')
    ax2.plot(data['testLoss'], 'x-', label='testLoss')
    ax2.set_ylabel('Loss')
    ax2.tick_params('y')

    # Add legends
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')

    # Show the plot
    plt.title("ResNet accuracy, test loss and train loss over 30 epochs")
    plt.legend()
    plt.savefig("5_2_resnet_over_epochs.png")

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

    try:
        with open("trainingdata.json", 'r') as json_file:
            dataTraining = json.load( json_file)
    except:
        pass
    
    plotdata(dataTraining)

