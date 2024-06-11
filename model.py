import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

"""
Defining, training, and evaluating the neural network model with PyTorch.
"""

# Define the dataset for the logic gates (XOR, AND, OR, NOR, NAND)
def get_dataset(gate):
    """Generate input-output pairs for the specified logic gate."""
    if gate == 'XOR':
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    elif gate == 'AND':
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)
    elif gate == 'OR':
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [1]], dtype=torch.float32)
    elif gate == 'NOR':
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[1], [0], [0], [0]], dtype=torch.float32)
    elif gate == 'NAND':
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[1], [1], [1], [0]], dtype=torch.float32)
    else:
        raise ValueError(f"Unknown gate: {gate}")
    return X, y

# Define the neural network model
class LogicGateNet(nn.Module):
    def __init__(self, hidden_units):
        """Initialize the neural network with one hidden layer."""
        super(LogicGateNet, self).__init__()
        self.fc1 = nn.Linear(2, hidden_units)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_units, 1)  # Second fully connected layer
    
    def forward(self, x):
        """Define the forward pass of the network."""
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the output of the first layer
        x = torch.sigmoid(self.fc2(x))  # Apply Sigmoid activation to the output of the second layer
        return x

def train_model(gate, learning_rate, hidden_units, epochs):
    """Train the neural network model on the specified logic gate dataset."""
    X, y = get_dataset(gate)  # Get dataset for the specified logic gate
    model = LogicGateNet(hidden_units)  # Initialize the model
    criterion = nn.BCELoss()  # Binary Cross Entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
    
    # Training loop
    for epoch in range(epochs):
        outputs = model(X)  # Forward pass
        loss = criterion(outputs, y)  # Compute loss
        optimizer.zero_grad()  # Zero the gradient buffers
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
    
    return model, loss.item()  # Return trained model and final loss

def evaluate_model(model, gate):
    """Evaluate the trained model on the specified logic gate dataset."""
    X, y = get_dataset(gate)  # Get dataset for the specified logic gate
    with torch.no_grad():
        outputs = model(X)  # Forward pass
        predicted = (outputs > 0.5).float()  # Convert outputs to binary predictions
        accuracy = (predicted == y).float().mean().item()  # Compute accuracy
    return accuracy, predicted.numpy()  # Return accuracy and predicted values

def plot_decision_boundary(model, gate):
    """Plot the decision boundary of the trained model for the specified logic gate."""
    X, y = get_dataset(gate)  # Get dataset for the specified logic gate
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid).reshape(xx.shape)  # Compute model output for each point in the grid
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.8)  # Plot the decision boundary
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap="coolwarm", edgecolors='k')  # Plot the original data points
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Decision Boundary')

