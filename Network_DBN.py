import torch
import torch.nn as nn
import numpy as np

# Define the pre-trained DBN model
class DBN(nn.Module):
    def __init__(self):
        super(DBN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(193, 512),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.layer6 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.layer7 = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.layer8 = nn.Linear(8, 50)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

# Load the pre-trained DBN model
dbn_model = DBN()
dbn_model.load_state_dict(torch.load('path/to/pretrained/dbn.pth'))

# Load the ESC-50 dataset
x_train = torch.from_numpy(np.load('path/to/esc50_train_features.npy'))
y_train = torch.from_numpy(np.load('path/to/esc50_train_labels.npy'))
x_test = torch.from_numpy(np.load('path/to/esc50_test_features.npy'))
y_test = torch.from_numpy(np.load('path/to/esc50_test_labels.npy'))

# Evaluate the pre-trained DBN model on the ESC-50 dataset
with torch.no_grad():
    dbn_model.eval()
    outputs = dbn_model(x_test.float())
    _, predicted = torch.max(outputs.data, 1)
    total = y_test.size(0)
    correct = (predicted == y_test).sum().item()
    accuracy = correct / total
    print('Accuracy of the pre-trained DBN model on the test set: {:.2f}%'.format(accuracy * 100))
