# Analysis and Comparison of Generative Models on Digits Datasets

## Project Objective

The goal of this project is to implement and analyze the performance of three machine learning models: **Restricted Boltzmann Machines (RBMs)**, **Deep Belief Networks (DBNs)**, and **Deep Neural Networks (DNNs)**.

We aim to explore the Binary AlphaDigits dataset to understand the effects of varying numbers of hidden units, training characters, and network layers (for DBNs) affect model performance for RBMs and DBNs. This parts concentrates on the unsupervised learning and generative capabilities of the models. With the MNIST dataset, we will then evaluate DNNs by comparing pre-trained models to non-pre-trained ones, and study the impact of critical hyperparameters such as the number of hidden neurons, training samples, and network depth on model accuracy. Finally, the project compares generative models, namely RBMs, DBNs, **Variational Autoencoders (VAEs)**, and **Generative Adversarial Networks (GANs)**, to assess their performance on the MNIST dataset.

## Contents

- `Etude_sur_Binary_alpha_digits.ipynb`: A notebook that contains experiments of generative models RBM and DBN on the Binary Alpha Digits dataset.

- `Etude_sur_MNIST.ipynb`: This notebook contains experiments on DNNs analyzing the influence of hyperparameters such as the number of hidden neurons, the number of training samples, and the network depth on the model's accuracy with the MNIST dataset.

- `Other_generative_models.ipynb`: A comparative study of generative models including RBMs, DBNs, VAEs, and GANs on the MNIST dataset.

- `gan.py`: Classes and functions for implementation of the GAN used in the `Other_generative_models.ipynb` notebook.

- `vae.py`: Classes and functions for implementation of the VAE used in the `Other_generative_models.ipynb` notebook.

- `principal_RBM_alpha.py`: Functions for initializing and training a RBM, as well as for generating images.

- `principal_DBN_alpha.py`: Functions for initializing and training a DBN, as well as for generating images.

- `principal_DNN_MNIST.py`: Functions for initializing, pretraining, and training a DNN on the MNIST dataset.

- `utils.py`: Utility functions.

## Dependencies

- Python 3.9
- PyTorch (for VAE, GAN models)
- Matplotlib 
- NumPy 

## Installation

Clone this repository using:

```
git clone https://github.com/wiamadnan/Deep-Learning.git
```
