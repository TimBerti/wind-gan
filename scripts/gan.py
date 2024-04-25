import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, series, channels):
        super(ResidualBlock, self).__init__()
        self.series = series
        self.conv_1 = nn.Conv2d(channels, channels, kernel_size=(series, 5), padding='same')
        self.conv_2 = nn.Conv2d(channels, channels, kernel_size=(series, 5), padding='same')
        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        x_ = x
        x = F.relu(self.conv_1(x))
        x = self.conv_2(x)
        x = self.batch_norm(x)
        x = x + x_
        return x

class Generator(nn.Module):
    def __init__(self, series):
        super(Generator, self).__init__()
        self.series = series
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(series, 5), padding='same')
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(series, 5), padding='same')
        self.conv3 = nn.Conv2d(8, 4, kernel_size=(series, 5), padding='same')
        self.conv4 = nn.Conv2d(4, 1, kernel_size=(series, 5), padding='same')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.sigmoid(self.conv4(x))
        return x[:,:,:,48:-48]

class Discriminator(nn.Module):
    def __init__(self, series):
        super(Discriminator, self).__init__()
        self.series = series
        self.pool = nn.MaxPool2d((1, 4), stride=(1, 4))
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(series, 5), padding='same')
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(series, 5), padding='same')
        self.fc1 = nn.Linear(8*6*series, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8*6*self.series)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x