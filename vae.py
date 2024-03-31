import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import numpy as np
import matplotlib.pyplot as plt

class VAE(torch.nn.Module ):
  def __init__(self, x_dim, h_dim1, h_dim2, z_dim, n_rows, n_cols, n_channels):
    super(VAE, self).__init__()

    self.n_rows = n_rows
    self.n_cols = n_cols
    self.n_channels = n_channels
    self.n_pixels = (self.n_rows)*(self.n_cols)
    self.z_dim = z_dim

    # encoder part
    self.fc1 = nn.Linear(x_dim, h_dim1)
    self.fc2 = nn.Linear(h_dim1, h_dim2)
    self.fc31 = nn.Linear(h_dim2, z_dim)
    self.fc32 = nn.Linear(h_dim2, z_dim)
    # decoder part
    self.fc4 = nn.Linear(z_dim, h_dim2)
    self.fc5 = nn.Linear(h_dim2, h_dim1)
    self.fc6 = nn.Linear(h_dim1, x_dim)

  def encoder(self, x):
    h = F.relu(self.fc1(torch.flatten(x, start_dim=1)))
    h = F.relu(self.fc2(h))
    return self.fc31(h), self.fc32(h)
  def decoder(self, z):
    h = F.relu(self.fc4(z))
    h = F.relu(self.fc5(h))
    y = torch.sigmoid(self.fc6(h))
    return y.view(-1, self.n_channels, self.n_rows, self.n_cols)

  def sampling(self, mu, log_var):
    # this function samples a Gaussian distribution, with average (mu) and standard deviation specified (using log_var)
    std = torch.exp(0.5*log_var) # Standard deviation
    eps = torch.randn_like(std)  # Epsilon ~ N(0, 1)
    return eps.mul(std).add_(mu) # return z sample

  def forward(self, x):
    z_mu, z_log_var = self.encoder(x)
    z = self.sampling(z_mu, z_log_var)
    return self.decoder(z), z_mu, z_log_var

  def loss_function(self,x, y, mu, log_var, beta=1):
    reconstruction_error = F.binary_cross_entropy(y, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_error + beta*KLD, reconstruction_error, KLD


def generate_images_vae(vae_model, n_images=5):
    device = next(vae_model.parameters()).device
    epsilon = torch.randn(n_images, 1, vae_model.z_dim, device=device)
    imgs_generated = vae_model.decoder(epsilon)
    return imgs_generated.cpu().detach().numpy()  # Move to CPU for compatibility with display function


def display_images_vae(imgs, n_cols, filename='vae_generation.png', save=True):
    n_imgs = imgs.shape[0]  # Total number of images
    n_rows = np.ceil(n_imgs / n_cols).astype(int)  # Compute the number of rows needed

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    # Adjust axs to be a 2D array for consistent indexing
    if n_rows == 1 or n_cols == 1:
        axs = np.array(axs).reshape(n_rows, n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            img_idx = i * n_cols + j
            if img_idx < n_imgs:  # Check if the current index is less than the total number of images
                # Display the image in grayscale
                axs[i, j].imshow(imgs[img_idx, 0, :, :], cmap='gray')
                axs[i, j].axis('off')
            else:
                axs[i, j].axis('off')  # Hide axes if there's no image to display

    plt.tight_layout()
    if save: 
        plt.savefig(filename)
    plt.show()


def train_vae(vae_model, data_train_loader, vae_optimizer, epoch, device, beta=1, sample_interval=20):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(data_train_loader):
        vae_optimizer.zero_grad()

        data = data.to(device)
        y, z_mu, z_log_var = vae_model(data)
        loss_vae, reconstruction_loss, kld_loss = vae_model.loss_function(data, y, z_mu, z_log_var, beta=beta)
        loss_vae.backward()
        train_loss += loss_vae.item()
        vae_optimizer.step()

        if batch_idx % 100 == 0:
            print(f'''Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_train_loader.dataset)} ({100. * batch_idx / len(data_train_loader):.0f}%)]\tLoss: {loss_vae.item() / len(data):.6f} \tReconstruction Loss: {reconstruction_loss.item() / len(data):.6f} \tKLD: {kld_loss.item() / len(data):.6f}''')
            
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(data_train_loader.dataset):.4f}')

    if(epoch % sample_interval == 0):
        generated_imgs = generate_images_vae(vae_model, n_images=25)
        display_images_vae(generated_imgs, n_cols=5, save=False)