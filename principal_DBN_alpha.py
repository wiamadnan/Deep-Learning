import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio

from principal_RBM_alpha import init_RBM, train_RBM, entree_sortie_RBM, sortie_entree_RBM

def init_DBN(neurons):
    
    RBM_list = [init_RBM(neurons[i], neurons[i+1]) for i in range(len(neurons) - 1)]
    print(len(RBM_list[0]))
    return RBM_list

def train_DBN(RBM_list, images, n_epoch=10000, lr_rate=0.01, batch_size=50):

    losses = []
    new_RBM = []
    for W, a, b in RBM_list:     
        W, a, b, loss = train_RBM(images, W, a, b, epochs=n_epoch, learning_rate=lr_rate, batch_size=batch_size, verbose = False)
        new_RBM.append([W,a,b])

        _, images = entree_sortie_RBM(images, W, b)
        losses.append(loss)
    return RBM_list, losses

def generer_image_DBN(RBM_list, n_images, n_iter, shape=(20, 16)):
    fig, axs = plt.subplots(n_images // 5, 5, figsize=(10, 2 * (n_images // 5)))
    fig.patch.set_facecolor('black')

    for i in range(n_images):
        image = np.random.rand(RBM_list[0][0].shape[0]) < 0.5

        for _ in range(n_iter):
            for W, a, b in RBM_list:
                _, image = entree_sortie_RBM(image.reshape(1, -1), W, b)
            
            for W, a, b in reversed(RBM_list):
                _, image = sortie_entree_RBM(image, W, a)
        
        ax = axs[i // 5, i % 5]
        ax.imshow(image.reshape(shape), cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()