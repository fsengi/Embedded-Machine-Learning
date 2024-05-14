import json
import matplotlib.pyplot as plt


def plotdata(data:dict) -> None:
    plt.plot(data['Adadelta']['0.1'], 'x-',  label='Adadelta lr=0.1')
    plt.plot(data['Adagrad']['0.005'], 'o-',  label='Adagrad lr=0.005')
    plt.plot(data['Adam']['0.001'], 'x-',  label='Adam lr=0.001')
    plt.plot(data['RAdam']['0.001'], 'x-',  label='RAdam lr=0.001')
    plt.plot(data['Rprop']['0.005'], 'x-',  label='Rprop lr=0.005')
    plt.plot(data['SGD']['0.05'], 'x-',  label='SGD lr=0.05')
    plt.title("accuracy over epochs on CNN for different optimizers")
    plt.ylabel("accuracy in %")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("acc_optimizer_3_3.png")

with open("data.json", 'r') as json_file:
    data = json.load( json_file)


plotdata(data=data)
