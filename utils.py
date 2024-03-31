import requests, gzip, shutil
import json
import os
import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.preprocessing import OneHotEncoder
import pickle
import matplotlib.pyplot as plt


urls = {
    "train-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
}


def download_data():
    for file_name, url in urls.items():
        # Download the file
        response = requests.get(url)
        open(file_name, "wb").write(response.content)
        
        # Decompress the file
        with gzip.open(file_name, 'rb') as f_in:
            with open(file_name[:-3], 'wb') as f_out: # Remove .gz from the filename for the output
                shutil.copyfileobj(f_in, f_out)
    
    # Verify the files
    print("Downloaded files:")
    for file_name in os.listdir('.'):
        if "ubyte" in file_name:
            print(file_name)


def read_mnist():
    X_train, y_train = loadlocal_mnist(images_path='train-images-idx3-ubyte',
                         labels_path='train-labels-idx1-ubyte')
    X_test, y_test = loadlocal_mnist(images_path='t10k-images-idx3-ubyte',
                         labels_path='t10k-labels-idx1-ubyte')
    
    # On binarise les images
    X_train = np.where(X_train > 126, 1, 0)
    X_test = np.where(X_test > 126, 1, 0)
    
    # On encode les targets
    oh = OneHotEncoder()
    y_train = oh.fit_transform(y_train.reshape(-1,1)).toarray()
    y_test = oh.fit_transform(y_test.reshape(-1,1)).toarray()
    
    return X_train, X_test, y_train, y_test


def save_object(obj, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def plot_error_rates(error_rates_pretrained, error_rates_not_pretrained, xvalues, xlabel, save_path=None):
    """
    Plot error rates for pretrained and not pretrained models on train and test sets,
    with an option to save the plot. This version corrects the warnings related to line styles.

    Parameters:
    - error_rates_pretrained: Dictionary of error rates for pretrained models.
    - error_rates_not_pretrained: Dictionary of error rates for not pretrained models.
    - xlabel: Label for the x-axis.
    - save_path: Path to save the plot image. If None, the plot is not saved.
    """
    import matplotlib.pyplot as plt

    # Extract the error rates
    train_errors_pretrained = [values['train'] for values in error_rates_pretrained.values()]
    test_errors_pretrained = [values['test'] for values in error_rates_pretrained.values()]
    train_errors_not_pretrained = [values['train'] for values in error_rates_not_pretrained.values()]
    test_errors_not_pretrained = [values['test'] for values in error_rates_not_pretrained.values()]

    # Plot fig
    plt.figure(figsize=(12, 8))

    plt.plot(xvalues, train_errors_pretrained, 'o-', label='Pretrained - Train', linewidth=2, markersize=8, color='navy')
    plt.plot(xvalues, test_errors_pretrained, 'o--', label='Pretrained - Test', linewidth=2, markersize=8, color='skyblue')
    plt.plot(xvalues, train_errors_not_pretrained, 's-', label='Not Pretrained - Train', linewidth=2, markersize=8, color='darkgreen')
    plt.plot(xvalues, test_errors_not_pretrained, 's--', label='Not Pretrained - Test', linewidth=2, markersize=8, color='lightgreen')

    plt.title('DNN Performance: Pretrained vs Not Pretrained', fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Error Rate', fontsize=14)
    plt.xticks(xvalues, labels=xvalues)  # Ensure x-ticks match the number of hidden layers exactly
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Add the performance difference
    for x, y1, y2 in zip(xvalues, test_errors_pretrained, test_errors_not_pretrained):
        plt.annotate(f'Δ={(y2-y1):.2f}', (x, (y1+y2)/2), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='red')

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved as {save_path}")
    plt.show()


def save_dict_to_json(dct, file_path):
    with open(file_path, 'w') as f:
        json.dump(dct, f, indent=4)


def display_images(imgs, n_cols, filename='dbn_generation.png', save=True, size=(28, 28)):
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
                axs[i, j].imshow(imgs[img_idx, :].reshape(size), cmap='gray')
                axs[i, j].axis('off')
            else:
                axs[i, j].axis('off')  # Hide axes if there's no image to display

    plt.tight_layout()
    if save: 
        plt.savefig(filename)
    plt.show()