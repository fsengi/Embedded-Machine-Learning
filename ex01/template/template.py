# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch

# Convinience functions
def plot_model(model=None, name='default'):
    # Visualize data
    plt.plot(torch.linspace(0, 1, 1000), ground_truth_function(torch.linspace(0, 1, 1000)), label='Ground truth')
    plt.plot(x_train, y_train, 'ob', label='Train data')
    plt.plot(x_test, y_test, 'xr', label='Test data')
    # Visualize model
    if model is not None:
        plt.plot(torch.linspace(0, 1, 1000), model(torch.linspace(0, 1, 1000)), label=f'Model of degree: {model.degree()}')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(f'{name}.png')
    plt.show()

# Generate data
n_samples = 11
noise_amplitude = 0.15

def ground_truth_function(x):
    # Generate data of the form sin(2 * Pi * x)
    # ---- Fill in the following:
    result = torch.sin(2 * np.pi * x)
    return result

torch.manual_seed(42)

x_test = torch.linspace(0, 1, n_samples)
y_test = ground_truth_function(x_test) + torch.normal(0., noise_amplitude, size=(n_samples,))
x_train = torch.linspace(0, 1, n_samples)
y_train = ground_truth_function(x_train) + torch.normal(0., noise_amplitude, size=(n_samples,))

# Test plotting
plot_model(name="testdata")

# Model fitting

def error_function(model, x_data, y_data):
    y_pred = model(x_data)
    # ---- Fill with the error function from the lecture
    error = torch.sqrt(torch.sum(torch.square(y_pred - y_data)))
    return error

model_degree = 3
test_err = []
train_err = []

model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)
train_err.append(error_function(model, x_train, y_train))
test_err.append(error_function(model, x_test, y_test))

print(f"{train_err=}, {test_err=}")

# Result plotting
plot_model(model, name="Initial_fit")

# ---- Continue with the exercises on the degree of the polynomial and the exploration of data size
def plot_Error(train_err, test_err, name='Errormetrics'):
    # Visualize data
    plt.plot(train_err, 'o-', label='Train error')
    plt.plot(test_err, 'x-', label='Test error')
    plt.title("RMS vs degree of polynomial")
    plt.xlabel("degree polynomial")
    plt.ylabel("RMS")
    plt.legend()
    plt.savefig(f'{name}.png')
    plt.show()

# degree of poynomial vs RMS
train_err = []
test_err = []
for mod_degree in range(0, 12):
    model = np.polynomial.Polynomial.fit(x_train, y_train, deg=mod_degree)
    train_err.append(error_function(model, x_train, y_train))
    test_err.append(error_function(model, x_test, y_test))
plot_Error(train_err, test_err)

# -------------------- vary sample size ------------------------------
# Generate data

model_degree = 10
n_samples = 10
target_error = 0.0001

while True:    
    noise_amplitude = 0.15

    x_test = torch.linspace(0, 1, n_samples)
    y_test = ground_truth_function(x_test) + torch.normal(0., noise_amplitude, size=(n_samples,))
    x_train = torch.linspace(0, 1, n_samples)
    y_train = ground_truth_function(x_train) + torch.normal(0., noise_amplitude, size=(n_samples,))

    model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)
    train_err = error_function(model, x_train, y_train)
    test_err = error_function(model, x_test, y_test)
    error = torch.abs(train_err-test_err)

    if error > target_error:
        n_samples = n_samples + 10
    else: 
        print(f"sample size should be around {n_samples}")
        break

# Test plotting
plot_model(model, name=f"error_smaller_{target_error}")
