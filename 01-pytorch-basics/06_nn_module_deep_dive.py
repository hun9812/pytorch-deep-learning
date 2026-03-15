import torch
import torch.nn as nn
import torch.optim as optim

# sub module
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x

# main module
class ComplexModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # feature extraction (3 -> 16 -> 16)
        features = self.feature_extractor(x)
        # classification (16 -> 1)
        output = self.classifier(features)
        return output
    
# example usage
# model instance (input_size=3, hidden_size=16, output_size=1)
model = ComplexModel(input_size=3, hidden_size=16, output_size=1)


x_train = torch.randn(100,3)
y_train = (x_train[:, 0:1] + 2*x_train[:, 1:2] - x_train[:, 2:3] + 0.5)

criterion = nn.MSELoss()

# Adam = momentum + RMSProp
# momentum: helps accelerate gradients vectors in the right directions, thus leading to faster converging.
# RMSProp: divides the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight.
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# change the model to training mode
model.train()
epochs = 500

for epoch in range(epochs):
    prediction = model(x_train)
    loss = criterion(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# no train anymore (no dropout or something else)
model.eval()
with torch.no_grad():
    test_input = torch.randn(1, 3)
    real_answer = (test_input[:, 0:1] + 2*test_input[:, 1:2] - test_input[:, 2:3] + 0.5)

    prediction = model(test_input)



print(f"Test Input: {test_input.numpy()}")
print(f"Real Answer: {real_answer.item():.4f}")
print(f"Predictions:{prediction.item():.4f}")