import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Hyperparameters
trainging_epochs = 15
batch_size = 100

# MNIST dataset
mnist_train = dsets.MNIST(root = './data/', train = True, transform = transforms.ToTensor(), download = True)
mnist_test = dsets.MNIST(root = './data/', train = False, transform = transforms.ToTensor(), download = True)

# Data Loader (Input Pipeline)
# shuffle = True, shuffle the data on every epoch to prevent overfitting
# drop_last, drop the last batch if it's not full
data_loader = DataLoader(dataset = mnist_train, batch_size = batch_size, shuffle = True, drop_last = True)

# model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
# 28* 28 = 784 -> 256 -> 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(trainging_epochs):
    avg_loss = 0
    total_batch = len(data_loader)

    # x(input data), y(label/target)
    # x.shape = [batcgh_size, 1, 28, 28] -> view(-1, 28*28) -> [batch_size, 784]
    for x,y in data_loader:
        x = x.view(-1, 28*28)

        optimizer.zero_grad()
        hypothesis = model(x)
        loss = criterion(hypothesis, y)
        loss.backward()
        optimizer.step()

        avg_loss += loss / total_batch
    
    print((f'Epoch: {epoch+1}/{trainging_epochs}, Loss: {avg_loss:.4f}'))

# Test the model
model.eval()
with torch.no_grad():
    x_test = mnist_test.data.view(-1, 28*28).float()
    y_test = mnist_test.targets

    prediction = model(x_test)
    # torch.argmax(prediction, 1) return the index of the maximum value in each row
    correct_prediction = torch.argmax(prediction, 1) == y_test
    accuracy = correct_prediction.float().mean()

    print(f'Accuracy: {accuracy:.4f}')