import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio

def lire_alpha_digit(data_path, caracters_idx):
    """
    Load and preprocess images from the Binary AlphaDigits dataset
    
    Args:
        data_path (str): Path to the dataset file
        characters_idx (list of int): Indices of characters to load
    
    Returns:
        np.ndarray: Flattened images of the specified characters
    """
    alpha_digit = sio.loadmat(data_path)['dat']

    imgs_set = alpha_digit[caracters_idx,:].flatten()
    imgs = []

    for img in imgs_set:
        imgs.append(img.flatten())

    imgs_set_flatten = np.array(imgs)

    return imgs_set_flatten

def sigmoid(x):
    """
    Compute the sigmoid function
    
    Args:
        x (np.ndarray): Input array
    
    Returns:
        np.ndarray: Sigmoid of the input
    """
    return 1 / (1 + np.exp(-x))

def init_RBM(p, q):
    """
    Initialize the weights and biases of an RBM
    
    Args:
        p (int): Number of visible units
        q (int): Number of hidden units
    
    Returns:
        tuple: Weights, visible biases, and hidden biases
    """
    W = np.random.normal(loc=0, scale=0.01, size=(p, q))
    a = np.zeros(p)
    b = np.zeros(q)

    return W, a, b

def entree_sortie_RBM(X, W, b):
    """
    Calculate the activation of hidden units
    
    Args:
        X (np.ndarray): Data input
        W (np.ndarray): Weights matrix
        b (np.ndarray): Hidden biases
    
    Returns:
        tuple: Probabilities and sampled hidden states
    """
    act = np.dot(X, W) + b
    proba_h = sigmoid(act)

    # h_s = 1 * (np.random.rand(*proba_h.shape) < proba_h)
    h_s = np.random.binomial(1, proba_h, size=proba_h.shape)
    return proba_h, h_s


def sortie_entree_RBM(Y, W, a):
    """
    Reconstruct the visible units
    
    Args:
        Y (np.ndarray): Hidden unit activations
        W (np.ndarray): Weights matrix
        a (np.ndarray): Visible biases
    
    Returns:
        tuple: Probabilities and sampled visible states
    """
    proj = np.dot(Y, W.T) + a
    proba_v = sigmoid(proj)

    # v_s = 1 * (np.random.rand(*proba_v.shape) < proba_v)
    v_s = np.random.binomial(1, proba_v, size=proba_v.shape)
    return proba_v, v_s

def train_RBM(images, W, a, b, epochs=10000, learning_rate=0.01, batch_size=50, verbose=True):
    """
    Train an RBM using the Contrastive Divergence algorithm
    
    Args:
        images (np.ndarray): Input data
        W, a, b: RBM parameters to be trained
        epochs (int): Number of epochs for training
        learning_rate (float): Learning rate
        batch_size (int): Size of minibatches
        verbose (bool): If True, print progress and loss
    
    Returns:
        tuple: Updated weights, visible biases, hidden biases, and loss history
    """
    n_samples = images.shape[0]
    loss = []

    for epoch in range(1, epochs + 1):
        img_shuffled = np.copy(images)
        np.random.shuffle(img_shuffled)

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            x_batch = img_shuffled[batch_start:batch_end, :]

            proba_h0, h0 = entree_sortie_RBM(x_batch, W, b)
            proba_v0, v1 = sortie_entree_RBM(h0, W, a)
            proba_h1, h1 = entree_sortie_RBM(v1, W, b)

            grad_w = np.dot(x_batch.T, proba_h0) - np.dot(v1.T, proba_h1)
            grad_a = np.mean(x_batch - v1, axis=0)
            grad_b =np.mean(proba_h0 - proba_h1, axis=0)
            
            W += learning_rate * (np.dot(x_batch.T, proba_h0) - np.dot(v1.T, proba_h1))
            a += learning_rate * np.mean(x_batch - v1, axis=0)
            b += learning_rate * np.mean(proba_h0 - proba_h1, axis=0)

        Y, _ = entree_sortie_RBM(img_shuffled, W, b)
        new_X, _ = sortie_entree_RBM(Y, W, a)
        loss_epoch = np.sum((new_X - img_shuffled) ** 2) / (n_samples * a.shape[0])
        loss.append(loss_epoch)

        if verbose:
            if not(epoch % 20) or epoch == 1:
                print(f'Epoch {epoch} out of {epochs}, loss: {loss[-1]}')

    return W, a, b, loss

def generer_image_RBM(n_imgs, n_iter, W, a, b, shape=(20, 16)):
    """
    Generate images from an RBM
    
    Args:
        n_imgs (int): Number of images to generate
        n_iter (int): Number of Gibbs sampling iterations
        W, a, b: RBM parameters
        shape (tuple): Shape of each image
    """
    fig, axs = plt.subplots(n_imgs // 5, 5, figsize=(10, 2 * (n_imgs // 5)))
    fig.patch.set_facecolor('black')

    generated_imgs = []
    
    for i in range(n_imgs):
        v = np.random.binomial(1, 0.5, size=W.shape[0])
        # v = np.random.rand(W.shape[0]) < 0.5

        for i in range(n_iter):
            _, h = entree_sortie_RBM(v.reshape(1, -1), W, b)
            _, v = sortie_entree_RBM(h, W, a)

        generated_imgs.append(v)
        
        ax = axs[i // 5, i % 5]
        ax.imshow(v.reshape(shape), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return np.array(generated_imgs)