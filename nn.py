# Neural Network functions
# @author: gbrolo

import numpy as np
import pandas as pd

CHECK_EPSILON = 0.0001  # epsilon for Gradient Checking
INIT_EPSILON = 0.12     # epsilon independent from Gradient Checking 

sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
sigmoidPrime = np.vectorize(lambda x: np.multiply(sigmoid(x), 1 - sigmoid(x)))
initializeWeights = np.vectorize(
    lambda L_input, L_output: np.random.rand(L_output, L_input + 1) * 2 * INIT_EPSILON - INIT_EPSILON
)

# feeds forward and then backpropagates to find D matrix, containing partial derivatives
def backpropagation(params, L_input_size, HL_output_size, classes, X, y, lmbda):    
    # initial thetas
    theta_1 = np.reshape(params[:HL_output_size*(L_input_size+1)], (HL_output_size, L_input_size+1), 'F')
    theta_2 = np.reshape(params[HL_output_size*(L_input_size+1):], (classes, HL_output_size+1), 'F')

    # hot encode y values
    y_hot_encoded = pd.get_dummies(y.flatten())

    # set deltas
    delta_1 = np.zeros(theta_1.shape)
    delta_2 = np.zeros(theta_2.shape)

    m = len(y)
    
    for i in range(X.shape[0]):
        ones = np.ones(1)
        a1 = np.hstack((ones, X[i]))
        z2 = a1 @ theta_1.T
        a2 = np.hstack((ones, sigmoid(z2)))
        z3 = a2 @ theta_2.T
        a3 = sigmoid(z3)

        d3 = a3 - y_hot_encoded.iloc[i,:][np.newaxis,:]
        z2 = np.hstack((ones, z2))
        d2 = np.multiply(
            theta_2.T @ d3.T, 
            sigmoidPrime(z2).T[:,np.newaxis]
        )
        delta_1 = delta_1 + d2[1:,:] @ a1[np.newaxis,:]
        delta_2 = delta_2 + d3.T @ a2[np.newaxis,:]
        
    delta_1 = delta_1 / m
    delta_2 = delta_2 / m
    
    delta_1[:,1:] = delta_1[:,1:] + theta_1[:,1:] * lmbda / m
    delta_2[:,1:] = delta_2[:,1:] + theta_2[:,1:] * lmbda / m
        
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

    ones = np.ones((m,1))
    a1 = np.hstack((ones, X))
    a2 = sigmoid(a1 @ theta_1.T)
    a2 = np.hstack((ones, a2))
    h = sigmoid(a2 @ theta_2.T)
    
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
def checkBackpropagation(
    params, backpropagation_params, L_input_size, HL_output_size, classes, X, y, lmbda=0.0
):         

    for i in range(5):
        x = int(np.random.rand() * len(params))
        eps_vector = np.zeros((len(params), 1))
        eps_vector[x] = CHECK_EPSILON

        cost_high = cost(params + eps_vector.flatten(), L_input_size, HL_output_size, classes, X, y, lmbda)
        cost_low  = cost(params - eps_vector.flatten(), L_input_size, HL_output_size, classes, X, y, lmbda)
        gradient = (cost_high - cost_low) / float(2 * CHECK_EPSILON)

        print("Random selection: {0} ; Real = {1:.6f} ; Backpropagation = {2:.6f}".format(x, gradient, backpropagation_params[x]))

# performs gradient descent iteration for recalculating theta (find theta opt)
def gradient_descent(L_input_size, HL_output_size, classes, X, y, theta, lmbda, iterations = 100):
    m = len(y)
    for i in range(iterations):
        theta -= backpropagation(theta, L_input_size, HL_output_size, classes, X, y, lmbda)

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
    total = reduce((lambda x, val: x + val), h[0])
    percentages = list(map(lambda x: (x * 100) / total, h[0]))
    return np.argmax(h, axis = 1), percentages
