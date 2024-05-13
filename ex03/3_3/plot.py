import json
import matplotlib.pyplot as plt


def plotdata(data:dict) -> None:
    plt.plot(data['Adadelta']['0.1'], 'x-',  label='Adadelta')
    plt.plot(data['Adagrad']['0.005'], 'o-',  label='Adagrad')
    plt.plot(data['Adam']['0.0001'], 'x-',  label='Adam')
    plt.plot(data['RAdam']['0.001'], 'x-',  label='RAdam')
    plt.plot(data['Rprop']['0.005'], 'x-',  label='Rprop')
    plt.plot(data['SGD']['0.05'], 'x-',  label='SGD')
    plt.title("accuracy over epochs on CNN for different optimizers")
    plt.ylabel("accuracy in %")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("acc_optimizer_3_3.png")

with open("data.json", 'r') as json_file:
    data = json.load( json_file)


plotdata(data=data)
