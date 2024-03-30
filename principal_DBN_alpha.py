import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio

from principal_RBM_alpha import init_RBM, train_RBM, entree_sortie_RBM, sortie_entree_RBM

def init_DBN(neurons):
    """
    Initialize a Deep Belief Network (DBN) with a sequence of Restricted Boltzmann Machines (RBMs)
    
    Args:
        neurons (list of int): A list of number of neurons in each layer of the DBN,
                               including the input layer and each hidden layer
    
    Returns:
        list: A list of initialized RBMs, where each RBM is represented as a tuple containing
              the weight matrix, visible biases, and hidden biases
    """
    DBN = [init_RBM(neurons[i], neurons[i+1]) for i in range(len(neurons) - 1)]
    return DBN

def train_DBN(DBN, images, epochs=100, learning_rate=0.01, batch_size=128, verbose=True):
    """
    Train a DBN using the provided images
    
    Args:
        DBN (list): A list of RBMs representing the DBN to be trained
        images (np.ndarray): An array of images used for training the DBN
        epochs (int, optional): The number of epochs to train each RBM
        learning_rate (float, optional): The learning rate for training
        batch_size (int, optional): The size of each mini-batch for trainin
        verbose (bool, optional): If True, prints progress and loss information
    
    Returns:
        tuple: A tuple containing the list of trained RBMs and the list of losses for each RBM
    """
    losses = []
    new_DBN = []
    for W, a, b in DBN:     
        W, a, b, loss = train_RBM(
            images=images,
            W=W,
            a=a,
            b=b, 
            epochs=epochs, 
            learning_rate=learning_rate, 
            batch_size=batch_size, 
            verbose=verbose
        )
        new_DBN.append([W,a,b])

        _, images = entree_sortie_RBM(images, W, b)
        losses.append(loss)
    return new_DBN, losses

def generer_image_DBN(DBN, n_images, n_iter, shape=(20, 16)):
    """
    Generate and display images from a trained DBN
    
    Args:
        DBN (list): A list of RBMs representing the trained DBN
        n_images (int): The number of images to generate
        n_iter (int): The number of Gibbs sampling iterations to perform for each image
        shape (tuple): The height and width to reshape each generated image for display
    
    Returns:
        np.ndarray: An array of generated images. Each image is represented as a flattened array of pixels
    """
    rows = n_images // 5 + int(n_images % 5 != 0)  # Calculate rows needed
    fig, axs = plt.subplots(rows, 5, figsize=(10, 2 * rows))
    fig.patch.set_facecolor('black')

    if n_images <= 5:
        axs = np.array([axs])  # Ensure axs is 2D even for n_images <= 5

    generated_images = []

    for i in range(n_images):
        image = np.random.binomial(1, 0.5, size=DBN[0][0].shape[0])
        
        for _ in range(n_iter):
            for W, a, b in DBN:
                _, image = entree_sortie_RBM(image.reshape(1, -1), W, b)
            
            for W, a, b in reversed(DBN):
                _, image = sortie_entree_RBM(image, W, a)
        
        generated_images.append(image)  # Add the final generated image to the list

        # Plotting the image
        ax = axs[i // 5, i % 5] if rows > 1 else axs[i % 5]
        ax.imshow(image.reshape(shape), cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

    return np.array(generated_images)