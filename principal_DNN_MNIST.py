import numpy as np
from scipy.special import softmax
from tqdm import tqdm

from principal_DBN_alpha import init_DBN, train_DBN, entree_sortie_RBM

def init_DNN(neurons):
    dnn = init_DBN(neurons)
    return dnn

def pretrain_DNN(DNN, data, epochs, learning_rate, batch_size, verbose=True):
    # Pretrain the DNN using the given data
    DNN[:len(DNN)-1], losses = train_DBN(
        RBM=DNN[:len(DNN)-1], 
        images=data, 
        n_epoch=epochs, 
        lr_rate=learning_rate, 
        batch_size=batch_size,
        verbose=verbose
    )
    return DNN, losses

def calcul_softmax(W, b, data):
    logits = np.dot(data, W) + b
    probs = softmax(logits, axis=1)
    return probs 

def entree_sortie_reseau(DNN, data):
    outputs = [data]
    for W, a, b in DNN[:-1]:  # Exclude the last layer for now
        probas, _ = entree_sortie_RBM(outputs[-1], W, b)
        outputs.append(probas)
    # Handle the last layer separately (classification layer)
    W, a, b = DNN[-1]
    softmax_output = calcul_softmax(W, b, outputs[-1])
    outputs.append(softmax_output)
    return outputs

def cross_entropy(predictions, targets):
    return -np.mean(targets * np.log(predictions + 1e-9))

def retropropagation(DNN, X, y, epochs, learning_rate , batch_size , verbose = True):
    losses = []
    
    n = X.shape[0]
    l = len(DNN)
    for epoch in range(epochs):
        
        loss_batches = []
        
        X_copy, y_copy = np.copy(X), np.copy(y)
        shuffle = np.random.permutation(n)
        X_copy, y_copy = X_copy[shuffle], y_copy[shuffle]
        
        for batch in tqdm(range(0, n, batch_size)):
            X_batch = X_copy[batch:min(batch+batch_size, n)]
            y_batch = y_copy[batch:min(batch+batch_size, n), :]

            tb = X_batch.shape[0]

            pred = entree_sortie_reseau(DNN, X_batch)
            loss_batches.append(cross_entropy(y_batch, pred[-1]))

            for i in range(l):
                if i==0:
                    delta = pred[l] - y_batch
                else :
                    delta = grad_a*(pred[l-i]*(1-pred[l-i]))
                
                W, a, b = DNN[l-i-1]
                
                grad_W = 1/tb * pred[l-i-1].T.dot(delta)
                grad_b = 1/tb * np.sum(delta, axis=0)
                grad_a = delta.dot(W.T)

                W -= learning_rate*grad_W
                b -= learning_rate*grad_b
                
                # Update the DNN parameters
                DNN[l-i-1] = (W, a, b)
                
        losses.append(np.mean(loss_batches))
        
        if verbose:
            # if not(epoch % 20):
            loss = []
            
            print(f"Epoch {epoch} out of {epochs}, loss: {losses[-1]}")
            
    return DNN, losses
    
def test_DNN(DNN, data, labels, verbose=True):
    preds = entree_sortie_reseau(DNN, data)
    preds = np.argmax(preds[-1], axis=1)
    labels = np.argmax(labels, axis=1)
    good_labels = np.sum(preds == labels)
    false_rate = (1 - good_labels / labels.shape[0])
    if verbose:
        print(f"The percentage of false labeled data is {false_rate}")
    return false_rate