import torch
import torch.nn as nn
import torch.optim as optim

# make some data (y = 2x + 1)
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[3], [5], [7]])

model = nn.Linear(1,1) # input size = 1, output size = 1

# set Optimizer & loss function
# SGD: Stochastic Gradient Descent
# MSELoss: Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# training loop
epochs = 2000
for epoch in range(epochs):
    
    # hypothesis
    prediction = model(x_train)

    # loss
    loss = criterion(prediction, y_train)

    # optimizer grad
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# results
print("-" * 50)
new_var = torch.FloatTensor([[4.0]])
pred_y = model(new_var)
print(f"Prediction after training: f(4) = {pred_y.item():.4f}")