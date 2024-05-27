import json
import matplotlib.pyplot as plt


def plot_test_acc_for_several_models(data:dict) -> None:
    for key in data.keys():
        _data = data[key]
        plt.plot(_data['accTest'], 'o-', label=str(key))


    # Show the plot
    plt.title("VGG11 test accuracy over 30 epochs")
    plt.legend()
    plt.show()
    plt.savefig("5_1_all_models.png")
    

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

if __name__ == "__main__":

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


    plot_test_acc_for_several_models(data=dataGPU)

    models = ["VGG11", "VGG11_BN", "VGG11_GN(1)", "VGG11_GN(16)", "VGG11_GN(32)"]

    for model in models:
        print(model)
        plot_one_model(dataGPU[model],30,model)




