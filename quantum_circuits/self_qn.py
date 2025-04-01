import os
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras import regularizers
from qiskit_aer import Aer
import numpy as np
import torch
import torch.nn
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
from sklearn.model_selection import train_test_split


def preprocessing(dataset, features, target):
    data = pd.read_csv(dataset)
    data.replace(to_replace = 0.0, value= np.nan,inplace=True)
    data = data.dropna()
    x = data[features]
    y = data[target]
    
    return x, y

class classical:
    def __init(self, x_input, y_input ):
        learnrate = 0.002556
        self.x_input = x_input
        self.yinput = y_input
        self.learnrate = learnrate
        
    def pre_q_layer()