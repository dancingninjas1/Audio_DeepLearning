import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchaudio

import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torchvggish
from torchvggish import vggish, vggish_input

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the VGGish model
model = vggish().to(device)
# Load the ESC-50 dataset
data_path = "C:/Users/apurv/Documents/GitHub/DeepLearning/ESC-50"
metadata = pd.read_csv(data_path + "/meta/esc50.csv")

# Encode labels
le = LabelEncoder()
metadata["target"] = le.fit_transform(metadata["category"])

# Split into training and validation sets
train_metadata, val_metadata = train_test_split(metadata, test_size=0.2, stratify=metadata["category"], random_state=42)


# Define a PyTorch dataset to load the audio samples
class ESC50Dataset(Dataset):
    def __init__(self, audio_files, labels):
        self.audio_files = audio_files
        self.labels = labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.audio_files[idx])
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform).squeeze(0)
        return mel_spectrogram, self.labels[idx]


# Split the data into training and testing sets
train_data, test_data = train_test_split(train_metadata, test_size=0.2, random_state=42)

# Create data loaders for the training and testing sets
train_loader = DataLoader(ESC50Dataset(train_data), batch_size=64, shuffle=True)
test_loader = DataLoader(ESC50Dataset(test_data), batch_size=64, shuffle=False)

# Define the training and validation datasets and data loaders
train_dataset = ESC50Dataset(train_metadata)
val_dataset = ESC50Dataset(val_metadata)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Define the model
class VGGishClassifier(nn.Module):
    def __init__(self):
        super(VGGishClassifier, self).__init__()
        self.features = hub.load('https://tfhub.dev/google/vggish/1')
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 50),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.from_numpy(x.numpy())  # add this line to convert the Numpy array to a PyTorch tensor
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


#model = VGGishClassifier().to(device)
model = torch.hub.load('harritaylor/torchvggish', 'vggish')

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
total_steps = len(train_dataset)
# Train the model
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):

        # Move tensors to the configured device
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                     loss.item()))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
