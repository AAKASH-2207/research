import pennylane as qml
import torch
import torch.nn as nn

n_qubits = 4

dev = qml.device("default.qubit", wires = n_qubits)
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RX(inputs[i], wires = i)
        
    for i in range(n_qubits):
        qml.RY(weights[i],wires = i)
        
    
    return [qml.expval(qml.PauliZ[i]) for i in range(n_qubits)]

class hybridquantumneuralnetwork(nn.Module):
    def __init(self, n_qubits, n_classical_units):
        #super(hybridquantumneuralnetwork, self).__init__()
        self.n_qubits = n_qubits
        self.quantum_layer = quantum_circuit
        self.classical_layer = nn.Linear(n_qubits, n_classical_units)
        self.output_layer = nn.Linear(n_classical_units, 1)
        
    def forward(self, x):
        x = torch.tanh(x)
        
        weights = torch.randn(self.n_qubits)
        x = torch.tensor([self.quantum_layer(x, weights)], requires_grad = True)
        
        x =self.classical_layer(x)
        x = torch.sigmoid(self.output_layer(x))
        
        return x

model = hybridquantumneuralnetwork(n_qubits = 4, n_classical_units = 8)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

for epoch in range(10):
    optimizer.zero_grad()
    inputs = torch.randn(4)
    
    outputs = model(inputs)
    target = torch.tensor([1.0])
    loss = criterion(outputs, target)
    
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')