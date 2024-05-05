# author: https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch

import itertools
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from data_pipeline import prepare_dataset
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns

input_dim= 2
hidden_dim = 10
output_dim = 1

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self). __init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform(self.layer_1.weight, nonlinearity = "relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))

        return x


model = NeuralNetwork(input_dim, hidden_dim ,output_dim)
print(model)

learning_rate = 0.1
loss_fcn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

num_epochs = 100
loss_vals = []

train_dataloader, test_dataloader = prepare_dataset()

for epoch in range(num_epochs):
    for x, y in train_dataloader:
        optimizer.zero_grad()

        # forward, backward, optimizer
        pred = model(x)
        loss = loss_fcn(pred, y.unsqueeze(-1))
        loss_vals.append(loss.item())
        loss.backward()
        optimizer.step()

print("Training complete...:)")

y_pred = []
y_test = []
total = 0
correct = 0

with torch.no_grad():
    for x, y in test_dataloader:
        outputs = model(x)
        predicted = np.where(outputs < 0.5, 0, 1)
        predicted = list(itertools.chain(*predicted))
        y_pred.append(predicted)
        y_test.append(y)
        total += y.size(0)
        correct += (predicted == y.numpy()). sum().item()

print(f'Accuracy of the network on the 3300 test instances: {100 * correct // total}%')

step = np.linspace(0, 100, 10500)
fig, ax = plt.subplots(figsize= (8, 5))
plt. plot(step, np.array(loss_vals))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt. ylabel("Loss")
plt.show()

