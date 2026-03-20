import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)


# can stack layers together using nn.Sequential
model = nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid()
)

optimizer = optim.SGD(model.parameters(), lr = 1)
# Binary Cross Entropy Loss
criterion = nn.BCELoss()

epochs = 1000
for epoch in range(epochs):
    # H(x) = Sigmoid(Wx + b)
    hypothesis = model(x_train)

    loss = criterion(hypothesis, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

with torch.no_grad():
    prediction = hypothesis >= torch.FloatTensor([0.5])
    print(prediction)
    print(y_train)