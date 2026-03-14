import torch
import torch.nn as nn
import torch.optim as optim

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# define model class
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__() # take nn.Module's function
        self.linear = nn.Linear(3,1)
    
    def forward(self, x):
        # how the data will pass through the model
        return self.linear(x)
    

model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr = 1e-5)
criterion = nn.MSELoss()

epochs = 5000
for epoch in range(epochs):
    # forward
    prediction = model(x_train)

    # loss
    loss = criterion(prediction, y_train)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print("-" * 50)
print(f"Prediction after training: f(73, 80, 75) = {pred_y.item():.4f}")

paramameters = list(model.parameters())
print(f"Learned Weights (w): \n{paramameters[0]}")
print(f"Learned Bias (b): \n{paramameters[1]}")