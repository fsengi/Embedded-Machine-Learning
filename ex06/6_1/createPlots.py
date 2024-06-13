import matplotlib.pyplot as plt
import json
    
if __name__ == '__main__':
    with open('data_no_prune.json') as f:
        data_no_prune = json.load(f)
    with open('data_structured_10.json') as f:
        data_structured_10 = json.load(f)
    with open('data_structured_20.json') as f:
        data_structured_20 = json.load(f)
    with open('data_structured_50.json') as f:
        data_structured_50 = json.load(f)
    with open('data_structured_80.json') as f:
        data_structured_80 = json.load(f)
    with open('data_structured_90.json') as f:
        data_structured_90 = json.load(f)


    plt.plot(data_no_prune["accTest"], label='No_Prune')
    plt.plot(data_structured_10["accTest"], label='Structured_10%')
    plt.plot(data_structured_20["accTest"], label='Structured_20%')
    plt.plot(data_structured_50["accTest"], label='Unstructured_50%')
    plt.plot(data_structured_80["accTest"], label='Unstructured_80%')
    plt.plot(data_structured_90["accTest"], label='Unstructured_90%')
    plt.title("accuracy over epochs for pruning")
    plt.ylabel("accuracy in %")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("prune2_data.png")