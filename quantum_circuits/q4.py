from qiskit_aer import Aer
<<<<<<< HEAD
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


class quantumcircuit:
    def __init__(self, n_qubits, backend, shots):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')
        
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        
        self.backend = backend
        self.shots = shots
    def run(self, thetas):
        t_qc = transpile(self._circuit,self.backend)
        qobj = assemble(t_qc,
                        shots = self.shots,
                        parameter_binds=[{self.theta:theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys()))
        
        probabilities = counts / self.shots
        
        expectation = np.sum(states * probabilities)
        
        return np.array([expectation])
    
simulator = Aer.get_backend('aer_simulator')
circuit = quantumcircuit(1, simulator, 100)
print('expected values for rotation of pi {}'.format(circuit.run([np.pi])[0]))
circuit._circuit.draw()
=======
from qiskit import QuantumCircuit, transpile
import numpy as np
import torch
import torch.nn
>>>>>>> 0f76dd91ec88b2213be3bf0f15932c1b3f08a1f2
