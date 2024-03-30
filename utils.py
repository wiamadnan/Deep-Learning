import requests, gzip, shutil
import os
import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.preprocessing import OneHotEncoder
import pickle


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

def plot_results(x, y1, y2, xlabel, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label='Pre-trained + Trained', marker='o')
    plt.plot(x, y2, label='Trained Only', marker='x')
    plt.xlabel(xlabel)
    plt.ylabel('Error Rate')
    plt.title('DNN Performance by Layer Configuration')
    plt.legend()
    plt.savefig(filename)
    plt.show()