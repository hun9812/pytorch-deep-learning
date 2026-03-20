import torch
import torch.nn as nn
import torch.optim as optim

x_train = torch.FloatTensor([[1,2,1,1], [2,1,3,2], [3,1,3,4],
                              [4,1,5,5], [1,7,5,5], [1,2,5,6],
                              [1,6,6,6], [1,7,7,7]]
                              )
y_train = torch.LongTensor([2,2,2,1,1,1,0,0])

# 4 input, 3 output
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

# Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

epochs = 1000
for epoch in range(epochs):
    prediction = model(x_train)

    loss = criterion(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    test_data = torch.FloatTensor([[1,7,7,7]])
    pred = model(test_data)

    print(pred)
    print(torch.softmax(pred, dim = 1))
    print(pred.argmax(dim = 1))