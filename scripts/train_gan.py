import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import complete_auto_corr
from gan import Generator, Discriminator


REGIME = 1
BATCH_SIZE_G = 256
BATCH_SIZE_D = 512
MINIBATCH_SIZE = 32
SUB_EPOCHS = [(3, 3)] * 50
LEARNING_RATE_G = 1e-5
LEARNING_RATE_D = 1e-5
NOISE_DIM = 100
STAT_LOSS_WEIGHT = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    X = np.load('../data/X.npy')
    X = torch.tensor(X, dtype=torch.float32)
    y = np.load('../data/y.npy')
    y = torch.tensor(y, dtype=torch.float32)
    X = X[y == REGIME]
    X = (X - X.mean()) / X.std()
    X = X.reshape(-1, 1, 1, 96)
    return X

def stat_loss(fake_data, auto_corr, mean, std):
    fake_data = fake_data.reshape(-1, 1, 9, 96)
    mean_loss = torch.sum((fake_data.mean() - mean)**2)
    std_loss = torch.sum((fake_data.std() - std)**2)
    fake_auto_corr = complete_auto_corr(fake_data, 10)
    auto_corr_loss = torch.sum((fake_auto_corr.mean(axis=1) - auto_corr.mean(axis=1))**2)
    return mean_loss + std_loss + auto_corr_loss

def reset_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def train_discriminator(optimizer, model, criterion, X):
    optimizer.zero_grad()

    indices = np.random.choice(X.shape[0], MINIBATCH_SIZE * 9, replace=False)
    real_data = X[indices].reshape(-1, 1, 9, 96)
    real_data = real_data + 0.001 * torch.randn_like(real_data, device=device)
    D_real = model(real_data)
    D_real_loss = criterion(D_real, torch.ones_like(D_real))
    
    fake_data = G(torch.randn(MINIBATCH_SIZE, NOISE_DIM, 1, 1, device=device))[:,:,:,25:-25]
    D_fake = model(fake_data.detach())
    D_fake_loss = criterion(D_fake, torch.zeros_like(D_fake))

    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    optimizer.step()
    return D_real_loss, D_fake_loss

def train_generator(optimizer, discriminator, generator, criterion, auto_corr, mean, std):
    optimizer.zero_grad()

    fake_data = generator(torch.randn(MINIBATCH_SIZE, NOISE_DIM, 1, 1, device=device))[:,:,:,25:-25]
    G_stat_loss = stat_loss(fake_data, auto_corr, mean, std)

    output = discriminator(fake_data)
    G_disc_loss = criterion(output, torch.ones_like(output))

    G_loss = G_disc_loss + STAT_LOSS_WEIGHT * G_stat_loss
    G_loss.backward()
    optimizer.step()
    return G_stat_loss, G_disc_loss

def train(n_sub_epochs, num_epochs, X):
    auto_corr = complete_auto_corr(X, 10)
    mean = X.mean()
    std = X.std()
    epoch = 0
    best_score = float('inf')
    for sub_epoch in n_sub_epochs:
        reset_model(D)
        for _ in range(sub_epoch[0]):
            epoch += 1
            for _ in range(BATCH_SIZE_D):
                D_real_loss, D_fake_loss = train_discriminator(optimizer_D, D, criterion, X)
            print(f"Epoch {epoch}/{num_epochs}, D Real Loss: {D_real_loss.item()}, D Fake Loss: {D_fake_loss.item()}")

        for _ in range(sub_epoch[1]):
            epoch += 1
            for _ in range(BATCH_SIZE_G):
                G_stat_loss, G_disc_loss = train_generator(optimizer_G, D, G, criterion, auto_corr, mean, std)
            print(f"Epoch {epoch}/{num_epochs}, G Stat Loss: {G_stat_loss.item()}, G Disc Loss: {G_disc_loss.item()}")

        if G_stat_loss < best_score:
            best_score = G_stat_loss
            best_G = G
            print(f"Best score: {G_stat_loss}")
            torch.save(best_G.state_dict(), f'../models/best_generator_regime{REGIME}.pth')

    return best_G

if __name__ == '__main__':
    X = load_data().to(device)
    D = Discriminator().to(device)
    G = Generator(NOISE_DIM).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE_G)
    optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE_D)

    num_epochs = sum(sum(sub_epoch) for sub_epoch in SUB_EPOCHS)
    train(SUB_EPOCHS, num_epochs, X)
    torch.save(G.state_dict(), f'../models/final_generator_regime{REGIME}.pth')