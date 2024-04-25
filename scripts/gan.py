import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv2d(1, 3, kernel_size=(9, 5), padding='same')
        self.conv_2 = nn.Conv2d(3, 1, kernel_size=(9, 5), padding='same')
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, x):
        x_ = x
        x = F.relu(self.conv_1(x))
        x = self.conv_2(x)
        x = self.batch_norm(x)
        x = x + x_
        return x

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim # z_dim x 1 x 1
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 8, kernel_size=(1, 12)),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
            ) # 8 x 1 x 12
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=(1, 12), stride=(1, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
            ) # 16 x 1 x 45
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=(1, 12), stride=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
            ) # 32 x 1 x 144
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 1), stride=(3, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
            ) # 16 x 3 x 144
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=(3, 1), stride=(3, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
            ) # 8 x 9 x 144
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            ) # 1 x 9 x 146
        self.residual_blocks = nn.Sequential(*[ResidualBlock() for _ in range(5)])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.pool1 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.pool2 = nn.MaxPool2d((1, 4), stride=(1, 4))
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(9, 5), padding='same')
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(9, 5), padding='same')
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(9, 5), padding='same')
        self.fc1 = nn.Linear(32*6*9, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, 32*6*9)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x