import torch
import torch.nn as nn
import torch.optim as optim

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# H(x) = w1x1 + w2x2 + w3x3 + b
# 3 inputs , 1 output
model = nn.Linear(3, 1)

optimizer = optim.SGD(model.parameters(), lr = 1e-5)
criterion = nn.MSELoss()

epochs = 5000
for epoch in range(epochs):
    prediction = model(x_train)
    loss = criterion(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print("-" * 50)
print(f"Prediction after training: f(73, 80, 75) = {pred_y.item():.4f}")