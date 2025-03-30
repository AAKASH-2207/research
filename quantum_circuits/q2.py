import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

# Define the quantum device
dev = qml.device("qiskit.aer", wires=3)

# Define the quantum circuit
@qml.qnode(dev)
def quantum_circuit(params):
    # Apply rotations based on the parameters
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    qml.RZ(params[2], wires=0)
    qml.RX(params[0], wires=1)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=1)
    qml.RX(params[0], wires=2)
    qml.RY(params[1], wires=2)
    qml.RZ(params[2], wires=2)
    
    # Apply CNOT gates
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    qml.draw(quantum_circuit)
    # Return the expectation value of the first qubit
    return qml.expval(qml.PauliZ(0))

class QuantumLayer(nn.Module):
    def __init__(self, num_parameters):
        super(QuantumLayer, self).__init__()
        self.num_parameters = num_parameters
        self.params = nn.Parameter(torch.randn(num_parameters))

    def forward(self, x):
        # Run the quantum circuit for each input sample
        outputs = []
        for i in range(x.shape[0]):
            output = quantum_circuit(self.params)  # Use the parameters directly
            outputs.append(output)

        # Convert the list of outputs to a tensor and ensure the shape is [batch_size, 1]
        return torch.tensor(outputs, dtype=torch.float32).view(-1, 1)

class HybridQuantumCircuit(nn.Module):
    def __init__(self):
        super(HybridQuantumCircuit, self).__init__()
        self.fc1 = nn.Linear(8, 6)
        self.q_layer = QuantumLayer(num_parameters=3)
        self.fc2 = nn.Linear(1, 1)  # Output layer should match the output of the quantum layer

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
    #acc = np.mean(outputs == ytrain)
    # Backward pass
    loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters
    print(loss, epoch)
    #print(acc)
    if epoch % 10 == 0:  # Print loss every 10 epochs
        print(f'Epoch [{epoch}/{num_epoch}], Loss: {loss.item():.4f}')

# After training, you can evaluate the model or make predictions
model.eval()
with torch.no_grad():
    predictions = model(Xtrain)
    print("Predictions:", predictions)