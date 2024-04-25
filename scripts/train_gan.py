import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from utils import auto_corr
from gan import Generator, Discriminator

SERIES = 9
BATCH_SIZE = 256
MINIBATCH_SIZE = 32
SUB_EPOCHS = [(10, 10)] * 10
LEARNING_RATE_G = 1e-5
LEARNING_RATE_D = 1e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    X = np.load('../data/X.npy')
    X = torch.tensor(X, dtype=torch.float32)
    y = np.load('../data/y.npy')
    y = torch.tensor(y, dtype=torch.float32)
    X = X[y == 0]
    X = (X - X.min()) / (X.max() - X.min())
    X = X.reshape(-1, 1, 1, 96)
    return X

def train_discriminator(optimizer, model, criterion, X):
    optimizer.zero_grad()

    indices = np.random.choice(X.shape[0], MINIBATCH_SIZE * SERIES, replace=False)
    real_data = X[indices].reshape(-1, 1, SERIES, 96).to(device)
    real_data = real_data + 0.01 * torch.randn_like(real_data, device=device)
    fake_data = G(torch.randn(MINIBATCH_SIZE, 1, SERIES, 2*96, device=device))

    D_real = model(real_data)
    D_real_loss = criterion(D_real, torch.ones_like(D_real))
    
    D_fake = model(fake_data.detach())
    D_fake_loss = criterion(D_fake, torch.zeros_like(D_fake))

    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    optimizer.step()
    return D_loss

def train_generator(optimizer, discriminator, generator, criterion, auto_corrs):
    optimizer.zero_grad()

    fake_data = generator(torch.randn(MINIBATCH_SIZE, 1, SERIES, 2*96, device=device))
    auto_corrs_fake = [auto_corr(fake_data, i + 1).mean() for i in range(len(auto_corrs))]
    G_stat_loss = sum(torch.abs(auto_corrs_fake[i] - auto_corrs[i]) for i in range(len(auto_corrs)))

    output = discriminator(fake_data)
    G_disc_loss = criterion(output, torch.ones_like(output))

    G_loss = G_disc_loss + G_stat_loss
    G_loss.backward()
    optimizer.step()
    return G_loss, auto_corrs_fake

def train(n_sub_epochs, num_epochs, X):
    epoch = 0
    best_score = float('inf')
    auto_corrs = [auto_corr(X, i).mean() for i in range(1, 4)]
    for sub_epoch in n_sub_epochs:
        for _ in range(sub_epoch[0]):
            for _ in range(BATCH_SIZE):
                D_loss = train_discriminator(optimizer_D, D, criterion, X)
            epoch += 1
            print(f"Epoch {epoch}/{num_epochs}, D Loss: {D_loss.item()}")

        for _ in range(sub_epoch[1]):
            for _ in range(BATCH_SIZE):
                G_loss, auto_corrs_fake = train_generator(optimizer_G, D, G, criterion, auto_corrs)
            epoch += 1
            print(f"Epoch {epoch}/{num_epochs}, G Loss: {G_loss.item()}")

        print(f"Auto-correlations real: {auto_corrs}")
        print(f"Auto-correlations fake: {auto_corrs_fake}")
        score = sum(torch.abs(auto_corrs_fake[i] - auto_corrs[i]) for i in range(len(auto_corrs)))
        if score < best_score:
            best_score = score
            best_G = G
            print(f"Best score: {score}")

    return best_G

if __name__ == '__main__':

    X = load_data()
    G = Generator(SERIES).to(device)
    D = Discriminator(SERIES).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE_G)
    optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE_D)

    num_epochs = sum(sum(sub_epoch) for sub_epoch in SUB_EPOCHS)

    best_G = train(SUB_EPOCHS, num_epochs, X)

    torch.save(best_G.state_dict(), '../models/best_generator.pth')
    torch.save(G.state_dict(), '../models/final_generator.pth')

    best_G.eval()
    fake_data = best_G(torch.randn(2, 1, SERIES, 2*96, device=device)).reshape(-1, 96).cpu()
    for i in range(12):
        plt.plot(fake_data[i].detach().numpy(), color='red')
        plt.plot(X[i, 0].detach().numpy(), color='blue')
    plt.savefig('../images/best_generator.png')

    G.eval()
    fake_data = G(torch.randn(2, 1, SERIES, 2*96, device=device)).reshape(-1, 96).cpu()
    for i in range(12):
        plt.plot(fake_data[i].detach().numpy(), color='red')
        plt.plot(X[i, 0].detach().numpy(), color='blue')
    plt.savefig('../images/final_generator.png')