# Neural Network functions
# @author: gbrolo

import numpy as np
import pandas as pd
from functools import reduce

CHECK_EPSILON = 0.0001  # epsilon for Gradient Checking
INIT_EPSILON = 0.12     # epsilon independent from Gradient Checking 

sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
sigmoidPrime = np.vectorize(lambda x: np.multiply(sigmoid(x), 1 - sigmoid(x)))
initializeWeights = np.vectorize(
    lambda L_input, L_output: np.random.rand(L_output, L_input + 1) * 2 * INIT_EPSILON - INIT_EPSILON
)

def feed_forward(X, theta_1, theta_2, ones):
    # feed forward or forward propagation    
    a1 = np.hstack((ones, X))                               # a(1) = x
    z2 = a1 @ theta_1.T                                     # z(2) = theta(1) @ a(1)
    a2 = np.hstack((ones, sigmoid(z2)))                     # a(2) = g(z(2)), g -> sigmoid
    z3 = a2 @ theta_2.T                                     # z(3) = theta(2) @ a(2)
    a3 = sigmoid(z3)                                        # output activation: a(3) = h_theta(x) = g(z(3))

    return a1, z2, a2, z3, a3

def backpropagate(y, a3, z2, theta_2, ones):
    # calculate error, backpropagation
    d3 = a3 - y                                             # d(3) = a(3) - y
    z2 = np.hstack((ones, z2))
    d2 = np.multiply(                                       # error for hidden layer
        theta_2.T @ d3.T,                                   # d(2) = theta(2)T @ d(3) .* g'(z(2))
        sigmoidPrime(z2).T[:,np.newaxis]                    # column vector
    )

    return d3, z2, d2

# feeds forward and then backpropagates to find D matrix, containing partial derivatives
def backpropagation(params, L_input_size, HL_output_size, classes, X, y, lmbda):    
    # initial thetas
    theta_1 = np.reshape(params[:HL_output_size*(L_input_size+1)], (HL_output_size, L_input_size+1), 'F')
    theta_2 = np.reshape(params[HL_output_size*(L_input_size+1):], (classes, HL_output_size+1), 'F')

    # hot encode y values
    y_hot_encoded = pd.get_dummies(y.flatten())

    # set deltas
    delta_1 = np.zeros(theta_1.shape)                           # D1
    delta_2 = np.zeros(theta_2.shape)                           # D2

    m = len(y)
    
    for i in range(X.shape[0]):
        # feed forward or forward propagation
        ones = np.ones(1)
        a1, z2, a2, z3, a3 = feed_forward(X[i], theta_1, theta_2, ones)

        # calculate error, backpropagation
        d3, z2, d2 = backpropagate(
            y_hot_encoded.iloc[i,:][np.newaxis,:], 
            a3, 
            z2, 
            theta_2, 
            ones
        )
        
        # storing gradients
        delta_1 = delta_1 + d2[1:,:] @ a1[np.newaxis,:]         # D1 = D1 + a(1) @ d(1 + 1)
        delta_2 = delta_2 + d3.T @ a2[np.newaxis,:]             # D2 = D2 + d(2 + 1) @ a(2)T

    # actual gradients for neural network D1 and D2
    delta_1 = delta_1 / m
    delta_2 = delta_2 / m
    
    # regularization terms
    delta_1[:,1:] = delta_1[:,1:] + theta_1[:,1:] * lmbda / m   # D1 = (1/m)(D1 + lambda * theta(1))
    delta_2[:,1:] = delta_2[:,1:] + theta_2[:,1:] * lmbda / m   # D2 = (1/m)(D2 + lambda * theta(2))
        
    return np.hstack((delta_1.ravel(order='F'), delta_2.ravel(order='F')))

# cost function
def cost(params, L_input_size, HL_output_size, classes, X, y, lmbda):
    # initial thetas
    theta_1 = np.reshape(
        params[:HL_output_size * (L_input_size + 1)], 
        (HL_output_size, L_input_size + 1), 'F'
    )
    theta_2 = np.reshape(
        params[HL_output_size * (L_input_size + 1):], 
        (classes, HL_output_size + 1), 'F'
    )
    
    m = len(y)
    h = prediction(m, theta_1, theta_2, X)
    
    # hot encode y values
    y_hot_encoded = pd.get_dummies(y.flatten())

    # compute summations for each side of cost function sums
    cost_left_side = np.sum(
        np.sum(
            np.multiply(y_hot_encoded, np.log(h)) +
            np.multiply(1 - y_hot_encoded, np.log(1 - h))
        ) / (-m)
    )
    
    sum1_right = np.sum(
        np.sum(
            np.power(theta_1[:,1:], 2), 
            axis = 1
        )
    )
    
    sum2_right = np.sum(
        np.sum(
            np.power(theta_2[:,1:], 2), 
            axis = 1
        )
    )

    cost_right_side = (lmbda / (2*m)) * (sum1_right + sum2_right)
    return cost_left_side + cost_right_side

# compares numerical gradient taken with cost function and gradient taken using backpropagation
def checkGradient(
    params, backpropagation_params, L_input_size, HL_output_size, classes, X, y, lmbda = 0.0, iterations = 15
):         

    for i in range(iterations):
        x = int(np.random.rand() * len(params))
        eps_vector = np.zeros((len(params), 1))
        eps_vector[x] = CHECK_EPSILON

        cost_high = cost(params + eps_vector.flatten(), L_input_size, HL_output_size, classes, X, y, lmbda)
        cost_low  = cost(params - eps_vector.flatten(), L_input_size, HL_output_size, classes, X, y, lmbda)
        gradient = (cost_high - cost_low) / float(2 * CHECK_EPSILON)

        print("Random selection: {0} ; Real = {1:.6f} ; Backpropagation = {2:.6f}".format(x, gradient, backpropagation_params[x]))

# # performs gradient descent iteration for recalculating theta (find theta opt)
# def gradient_descent(L_input_size, HL_output_size, classes, X, y, theta, lmbda, iterations = 50):
#     m = len(y)
#     theta_history = np.zeros((iterations, 2))
#     gradient_check = True

#     for i in range(iterations):
#         gradient = backpropagation(theta, L_input_size, HL_output_size, classes, X, y, lmbda)

#         # check if gradient in bp is correct
#         if (gradient_check):
#             checkGradient(theta, gradient, L_input_size, HL_output_size, classes ,X, y, lmbda)
#             gradient_check = False

#         theta = theta - gradient
#         theta_history[i,:] = theta.T

#     return theta

# performs gradient descent iteration for recalculating theta (find theta opt)
def gradient_descent(L_input_size, HL_output_size, classes, X, y, theta, lmbda, iterations = 100):
    m = len(y)
    for i in range(iterations):
        theta = theta - backpropagation(theta, L_input_size, HL_output_size, classes, X, y, lmbda)

    return theta

def batch_gradient_descent(
    L_input_size, HL_output_size, classes, X, y, theta, lmbda, iterations = 10, batch_size = 10
):
    m = len(y)
    batches = int(m / batch_size)    

    for iteration in range(iterations):
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]

        for i in range(0, m, batch_size):
            try:
                X_i = X[i:i + batch_size]
                y_i = y[i:i + batch_size]

                # X_i = np.c_[np.ones(len(X_i)), X_i]

                theta = theta - backpropagation(theta, L_input_size, HL_output_size, classes, X_i, y_i, lmbda)    
            except:
                pass

    return theta

# calculates H given params m, theta_1, theta_2 and X
# m: len(y) or 1 if going to make a single classification
# theta_1, theta_2: numpy arrays for optimized thetas
# X: array of training images or array with single image element for classification
def prediction(m, theta_1, theta_2, X):
    ones = np.ones((m,1))
    a1 = np.hstack((ones, X))
    a2 = sigmoid(a1 @ theta_1.T)
    a2 = np.hstack((ones, a2))
    h = sigmoid(a2 @ theta_2.T)
    return h

# for training nn
def analize(theta_1, theta_2, X, y):
    m = len(y)
    h = prediction(m, theta_1, theta_2, X)
    return np.argmax(h, axis = 1)

# for classifying single image
def analize_single(theta_1, theta_2, X):
    h = prediction(1, theta_1, theta_2, X)  
    print('h is: ', h)
    # total = reduce((lambda x, val: x + val), h[0])
    # percentages = list(map(lambda x: (x * 100) / total, h[0]))
    percentages = list(map(lambda x: x * 100, h[0]))
    return np.argmax(h, axis = 1), percentages

def get_theta_percentage(theta1_opt, theta2_opt, X, y):
    pred = analize(theta1_opt, theta2_opt, X, y)
    return np.mean(pred == y.flatten()) * 100
