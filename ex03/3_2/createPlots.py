import matplotlib.pyplot as plt
import json
    
if __name__ == '__main__':
    with open('CNN.json') as f:
        cnn = json.load(f)
    with open('MLP.json') as f:
        mlp = json.load(f)

    plt.plot(cnn["time"], cnn["accuracy"], label='CNN')
    plt.plot(mlp["time"], mlp["accuracy"], label='MLP')
    plt.title("accuracy over executiontime for CNN and MLP models")
    plt.ylabel("accuracy in %")
    plt.xlabel("time in s")
    plt.legend()
    plt.savefig("timeplot_3_2.png")
    plt.clf()
    plt.plot(cnn["accuracy"], label='CNN')
    plt.plot(mlp["accuracy"], label='MLP')
    plt.title("accuracy over epochs for CNN and MLP models")
    plt.ylabel("accuracy in %")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("epochplot_3_2.png")