import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Generator(nn.Module):
  def __init__(self, z_dim, h_dim_1, h_dim_2, n_rows,n_cols,n_channels):
    super(Generator, self).__init__()
    self.n_rows = n_rows
    self.n_cols = n_cols
    self.n_channels = n_channels
    self.n_pixels = (self.n_rows)*(self.n_cols)
    self.h_dim_1 = h_dim_1
    self.h_dim_2 = h_dim_2
    self.z_dim = z_dim

    self.fc1 = nn.Linear(self.z_dim, self.h_dim_1)
    self.fc2 = nn.Linear(self.h_dim_1, self.h_dim_2)
    self.fc3 = nn.Linear(self.h_dim_2, self.n_pixels * self.n_channels)
    
  def forward(self, z):
    y = F.leaky_relu(self.fc1(z), 0.2)
    y = F.leaky_relu(self.fc2(y), 0.2)
    y = torch.tanh(self.fc3(y))
    y = y.view(-1, self.n_channels, self.n_rows, self.n_cols)
    return(y)


class Discriminator(nn.Module):
  def __init__(self, h_dim_2, h_dim_1, z_dim, n_rows, n_cols, n_channels):
    super(Discriminator, self).__init__()

    self.n_rows = n_rows
    self.n_cols = n_cols
    self.n_channels = n_channels
    self.n_pixels = (self.n_rows)*(self.n_cols)
    self.h_dim_1 = h_dim_1
    self.h_dim_2 = h_dim_2
    self.z_dim = z_dim
    
    self.fc1 = nn.Linear(self.n_pixels * self.n_channels, self.h_dim_2)
    self.fc2 = nn.Linear(self.h_dim_2, self.h_dim_1)
    self.fc3 = nn.Linear(self.h_dim_1, 1) 

  def forward(self, x):
    x = x.view(-1, self.n_pixels * self.n_channels)
    y = F.leaky_relu(self.fc1(x), 0.2)
    y = F.leaky_relu(self.fc2(y), 0.2)
    y = torch.sigmoid(self.fc3(y))
    return y


def loss_fn_gen(d_gen_data):
  loss_gen = -torch.mean(torch.log(d_gen_data)) # FILL IN CODE
  return loss_gen


def generate_images_gan(generator, z_dim, n_images=25):
    device = next(generator.parameters()).device
    z_random = torch.randn(n_images, 1, z_dim, dtype=torch.float, device=device)  # Adjust shape if necessary for your generator
    gen_imgs = np.transpose(generator(z_random).cpu().detach().numpy() , (0,2,3,1))
    gen_imgs = 0.5 * gen_imgs + 0.5
    return gen_imgs


def display_images_gan(imgs, n_cols, filename='gan_generation.png', save=True):
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
                axs[i, j].imshow(imgs[img_idx, :, :, :], cmap='gray')
                axs[i, j].axis('off')
            else:
                axs[i, j].axis('off')  # Hide axes if there's no image to display

    plt.tight_layout()
    if save: 
        plt.savefig(filename)
    plt.show()



