from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
import numpy as np
import torch
import torch.nn as nn

# Quantum simulator backend
backend = Aer.get_backend('statevector_simulator')

def run_quantum_circuit(params):
    qc = create_quantum_circuit(params)  # Ensure create_quantum_circuit is defined
    compiled_circuit = transpile(qc, backend)  # Transpile the circuit for the backend
    result = backend.run(compiled_circuit).result()  # Run the circuit
    statevector = result.get_statevector()  # Get the statevector
    return statevector  # Return the statevector directly

def create_quantum_circuit(params):
    qc = QuantumCircuit(3)
    qc.ry(params[0], 0)
    qc.ry(params[1], 0)
    qc.ry(params[2], 0)
    qc.ry(params[0], 1)
    qc.ry(params[1], 1)
    qc.ry(params[2], 1)
    qc.ry(params[0], 2)
    qc.ry(params[1], 2)
    qc.ry(params[2], 2)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.cx(0, 2)
    print(qc)
    return qc

def parameter_shift_circuit(params, shift=0.01):
    gradients = []
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += shift
        params_minus[i] -= shift
        
        output_plus = run_quantum_circuit(params_plus)
        output_minus = run_quantum_circuit(params_minus)
        
        gradient = (output_plus - output_minus) / (2 * shift)
        gradients.append(gradient)
    
    return torch.tensor(gradients)

class QuantumLayer(nn.Module):
    def __init__(self, num_parameters):
        super(QuantumLayer, self).__init__()
        self.num_parameters = num_parameters
        self.params = nn.Parameter(torch.randn(num_parameters))

    def forward(self, x):
        # Run the quantum circuit
        statevector = run_quantum_circuit(self.params.detach().numpy())
        return torch.tensor(statevector, dtype=torch.float32)
    def backward(self, x):
        # Compute the gradient of the quantum circuit
        gradients = parameter_shift_circuit(self.params.detach().numpy())
        return gradients
    
class HybridQuantumCircuit(nn.Module):
    def __init__(self):
        super(HybridQuantumCircuit, self).__init__()
        self.fc1 = nn.Linear(8, 6)
        self.q_layer = QuantumLayer(num_parameters=3)
        self.fc2 = nn.Linear(8, 1)  # Output layer should match the output of the quantum layer

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.q_layer(x)  # Quantum layer output
        x = self.fc2(x)      # Final output layer
        return x

# Create random dataset
Xtrain = torch.randn(100, 8)
ytrain = torch.randn(100, 1)

# Initialize the model, loss function, and optimizer
model = HybridQuantumCircuit()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epoch = 100
for epoch in range(num_epoch):
    model.train()
    optimizer.zero_grad()  # Clear previous gradients
    outputs = model(Xtrain)  # Forward pass
    loss = criterion(outputs, ytrain)  # Calculate loss
    # Backward pass
    loss.backward()  # Compute gradients
    print(loss)
    optimizer.step()  # Update parameters
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Test the model
xtest = torch.randn(20, 8)
ytest = model(xtest)
print(loss)

print(f"Predicted Values: {ytest}")