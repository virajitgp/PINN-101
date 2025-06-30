
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 5)
        self.hidden_layer2 = nn.Linear(5, 5)
        self.output_layer = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.tanh(self.hidden_layer1(x))
        x = torch.tanh(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x

# Define the physics-informed loss function
def physics_informed_loss(net, x, x_boundary):
    u = net(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    pde_residual = u_x - 2 * torch.cos(2 * x)
    
    u_boundary = net(x_boundary)
    boundary_residual = u_boundary - 0.0
    
    return torch.mean(pde_residual**2) + torch.mean(boundary_residual**2)

# Training data
x_data = torch.linspace(0, 2 * np.pi, 100).view(-1, 1)
x_data.requires_grad = True
x_boundary = torch.tensor([[0.0]], requires_grad=True)

# Create the network and optimizer
net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Training loop
for i in range(1000):
    optimizer.zero_grad()
    loss = physics_informed_loss(net, x_data, x_boundary)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss.item()}")

# Plot the results
x_test = torch.linspace(0, 2 * np.pi, 100).view(-1, 1)
u_pred = net(x_test).detach().numpy()
u_true = np.sin(2 * x_test.numpy())

# Train a standard neural network
y_data = torch.sin(2 * x_data).detach()
net_nn = Net()
optimizer_nn = torch.optim.Adam(net_nn.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for i in range(1000):
    optimizer_nn.zero_grad()
    y_pred = net_nn(x_data)
    loss = loss_fn(y_pred, y_data)
    loss.backward()
    optimizer_nn.step()

    if i % 100 == 0:
        print(f"NN Iteration {i}, Loss: {loss.item()}")

# Plot the results
u_pred_nn = net_nn(x_test).detach().numpy()

plt.figure(figsize=(10, 8))
plt.plot(x_test.numpy(), u_true, label="True solution")
plt.plot(x_test.numpy(), u_pred, label="PINN solution", linestyle="--")
plt.plot(x_test.numpy(), u_pred_nn, label="NN solution", linestyle=":")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("PINN vs. NN solution for du/dx = 2*cos(2x)")
plt.grid(True)
plt.show()
plt.savefig('pinn_vs_nn_solution.png')
