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
from qiskit import *
import random
from qiskit_nature.drivers.second_quantization.pyscfd import PySCFDriver
from qiskit_nature.mappers.second_quantization import ParityMapper, BravyiKitaevMapper, JordanWignerMapper
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import NumPyMinimumEigensolverFactory
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit_nature.algorithms import ExcitedStatesEigensolver
from qiskit_nature.algorithms import NumPyEigensolverFactory
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem

from qiskit.visualization import *
from sklearn.model_selection import train_test_split

def preprocessing(dataset, features, target):
    data = pd.read_csv(dataset)
    data.replace(to_replace = 0.0, value= np.nan,inplace=True)
    data = data.dropna()
    x = data[features]
    y = data[target]
    
    return x, y

def encode(num_qubits, bl):
    qr_m = QuantumRegister(num_qubits, 'qr_m')
    cr_nn = ClassicalRegister(num_qubits, 'cr_nn')
    qc = QuantumCircuit(qr_m, cr_nn)
    qc.h(qr_m)
    qc.ry(bl, qr_m)
    return qc
def quantum_circuit(params):
    pass
class classical(nn.Module):
    def __init__(self, x_input, y_input, learnrate = 0.0025443,activation = 'relu' ):
        pre_model = keras.Sequential()
        pre_model.add(keras.layers.dense(1024,activation = activation, regularizers = keras.regularizers.l2(0.00466)))
        pre_model.add(keras.layers.dropout(0.04))
        pre_model.add(keras.layers.dense(1024,activation = activation))
        pre_model.add(keras.layers.dropout(0.04))
        pre_model.add(keras.layers.dense(512,activation = activation))
        return pre_model.activation
        
    