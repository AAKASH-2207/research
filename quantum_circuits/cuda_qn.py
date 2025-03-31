import cudaq
from cudaq import spin
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import funtion
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn
import torchvision
from sklearn.model_selection import train_test_split

torch.manual_seed(22)
cudaq.set_random_Seed(44)

#device =  torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudaq.set_traget("qpp-cpu")

def prepare_data(target_digits, sample_count, test_size):
    transform = transforms.Compose(
        [transforms.Totensor(),
        transforms.Normalize((0.1307), (0.3081))])

    dataset = datasets.MNIST("./data",
                        train=True,
                        download=True,
                        transform = transform)
    idx = (dataset.target == target_digits[0] | dataset.targets == target_digits[1])
    dataset.data = dataset.data[idx]
    datasets.targets = datasets.targets[idx]
    subset_indices = torch.randperm(dataset.data.size(0))[:sample_count]
    x = dataset.data[subset_indices].float().unsqueeze(1).to(device)

    y = dataset.targets[subset_indices].to(device).float().to(device)

    # Relabel the targets as a 0 or a 1.
    y = torch.where(y == min(target_digits), 0.0, 1.0)

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_size /
                                                        100,
                                                        shuffle=True,
                                                        random_state=42)

    return x_train, x_test, y_train, y_test

# Classical parameters.

sample_count = 1000  # Total number of images to use.
target_digits = [5, 6]  # Hand written digits to classify.
test_size = 30  # Percentage of dataset to be used for testing.
classification_threshold = 0.5  # Classification boundary used to measure accuracy.
epochs = 1000  # Number of epochs to train for.

# Quantum parmeters.

qubit_count = 1
hamiltonian = spin.z(0)  # Measurement operator.
shift = torch.tensor(torch.pi / 2)  # Magnitude of parameter shift.

x_train, x_test, y_train, y_test = prepare_data(target_digits, sample_count,
                                                test_size)

# Plot some images from the training set to visualise.
if device != 'cpu':
    sample_to_plot = x_train[:10].to(torch.device('cpu'))
else:
    sample_to_plot = x_train[:10]

grid_img = torchvision.utils.make_grid(sample_to_plot,
                                        nrow=5,
                                        padding=3,
                                        normalize=True)
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis('off')
plt.show()
