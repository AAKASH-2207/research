import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from numpy import pi
from nnfs.datasets import vertical_data
import matplotlib.pyplot as plt

x, y = vertical_data(samples= 100, classes= 3)

plt.vlines(x,0 ,  ymax = y, bins = 100)
plt.show()