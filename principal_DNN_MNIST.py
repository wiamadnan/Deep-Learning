import numpy as np
from principal_DBN_alpha import init_DBN, train_DBN, entree_sortie_RBM

def init_DNN(neurons):
    dnn = init_DBN(neurons)
    return dnn

def pretrain_DNN(DNN, data, epochs, learning_rate, batch_size):
    # Pretrain the DNN using the given data
    RBM_list, losses = train_DBN(DNN, data, n_epoch=epochs, lr_rate=learning_rate, batch_size=batch_size)
    return RBM_list, losses

def calcul_softmax(weights, biases, data):
    logits = np.dot(data, weights) + biases
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probabilities

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
    return -np.sum(targets * np.log(predictions + 1e-9)) / predictions.shape[0]

def sigmoid_derivative(x):
    return x * (1 - x)

def retropropagation(DNN, data, labels, epochs, learning_rate, batch_size):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, data.shape[0], batch_size):
            # Mini-batch data and labels
            batch_data = data[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            # Forward pass
            activations = entree_sortie_reseau(DNN, batch_data)
            predictions = activations[-1]
            
            # Calculate cross-entropy loss
            loss = cross_entropy(predictions, batch_labels)
            total_loss += loss
            
            # Backward pass
            # Error in the output layer
            output_error = predictions - batch_labels
            delta = output_error
            
            # Propagate errors back through the network
            for layer_idx in reversed(range(len(DNN))):
                activations_prev_layer = activations[layer_idx]
                W, a, b = DNN[layer_idx]
                
                # Calculate the gradient
                W_gradient = np.dot(activations_prev_layer.T, delta) / batch_size
                b_gradient = np.mean(delta, axis=0)
                
                # Update weights and biases
                W -= learning_rate * W_gradient
                b -= learning_rate * b_gradient
                
                # If not the first layer, propagate the error backward
                if layer_idx > 0:
                    delta = np.dot(delta, W.T) * sigmoid_derivative(activations_prev_layer)
                    
                # Update the DNN parameters
                DNN[layer_idx] = (W, a, b)
                
        total_loss /= (data.shape[0] / batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")
        
    return DNN

def test_DNN(DNN, data, true_labels):
    predicted_probs = entree_sortie_reseau(DNN, data)[-1]  # Get the last layer's output
    predicted_labels = np.argmax(predicted_probs, axis=1)
    true_labels = np.argmax(true_labels, axis=1)
    error_rate = np.mean(predicted_labels != true_labels)
    print("Error rate:", error_rate)
    return error_rate


