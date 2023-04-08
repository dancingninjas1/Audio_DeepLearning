import librosa
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

from torchvision.transforms import RandomApply, Compose, GaussianBlur, RandomCrop, RandomHorizontalFlip, \
    RandomResizedCrop


# Load the ESC-50 dataset
def load_data():
    audio_files = []
    labels = []
    with open('ESC-50/meta/esc50.csv', 'r') as f:
        for line in f.readlines()[1:]:
            values = line.strip().split(',')
            filename = 'ESC-50/audio/' + values[0]
            label = int(values[2])
            audio_files.append(filename)
            labels.append(label)
    return audio_files, labels


# Preprocess the audio files
def preprocess(audio_files, labels):
    X = []
    y = []
    for audio_file, label in zip(audio_files, labels):
        # Load the audio file
        audio, sr = librosa.load(audio_file, sr=None, mono=True)
        # Resample the audio file to a fixed sampling rate
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        # Convert the audio file to Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
        # Convert the Mel spectrogram to log Mel spectrogram
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize the log Mel spectrogram
        norm_log_mel_spec = (log_mel_spec - np.min(log_mel_spec)) / (np.max(log_mel_spec) - np.min(log_mel_spec))
        # Append the normalized log Mel spectrogram and the label to X and y
        X.append(norm_log_mel_spec)
        y.append(label)
    # Split the preprocessed dataset into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train), np.array(y_val), np.array(y_test)


# Load the ESC-50 dataset
audio_files, labels = load_data()

# Preprocess the audio files
X_train, X_val, X_test, y_train, y_val, y_test = preprocess(audio_files, labels)


class AudioDataset(Dataset):
    def __init__(self, X, y, transforms=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transforms = transforms

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transforms:
            x = self.transforms(x)
        return x, y


class CustomTransform:
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy().squeeze()  # Convert to numpy array and remove singleton dimensions
            x = (x - x.min()) / (x.max() - x.min())  # Normalize to [0, 1]
            x = np.stack([x, x, x], axis=-1)  # Convert grayscale image to RGB format
            x = Image.fromarray((x * 255).astype(np.uint8), mode='RGB')  # Convert numpy array to PIL Image
            x = TF.to_grayscale(x)  # Convert RGB to grayscale
        return x


transform_train = transforms.Compose([
    CustomTransform(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


class AudioCNN(nn.Module):
    def __init__(self, n_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=512)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=512 * 4 * 4, out_features=8192)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=8192, out_features=n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Define the audio CNN model
model = AudioCNN(n_classes=50)
torch.manual_seed(42)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.01, verbose=True)

# Create audio datasets and data loaders
train_dataset = AudioDataset(X_train[:, np.newaxis], y_train, transforms=transform_train)
val_dataset = AudioDataset(X_val[:, np.newaxis], y_val)
test_dataset = AudioDataset(X_test[:, np.newaxis], y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train the audio CNN model
num_epochs = 50
for epoch in range(num_epochs):
    # Train the model
    model.train()
    for X, y in train_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for X, y in val_loader:
            output = model(X)
            loss = criterion(output, y)
            val_loss += loss.item() * X.size(0)
            _, predicted = torch.max(output, 1)
            val_correct += (predicted == y).sum().item()
    val_loss /= len(val_dataset)
    val_acc = val_correct / len(val_dataset)
    print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'
          .format(epoch + 1, num_epochs, val_loss, val_acc))
