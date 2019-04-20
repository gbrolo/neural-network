import numpy as np
import pandas as pd

sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))

def sigmoidPrime(x):
    return np.multiply(sigmoid(x), 1 - sigmoid(x))

def initializeWeights(L_input, L_output):
    eps = 0.12
    return np.random.rand(L_output, L_input + 1) * 2 * eps - eps

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
        d2 = np.multiply(theta_2.T @ d3.T, sigmoidPrime(z2).T[:,np.newaxis])
        delta_1 = delta_1 + d2[1:,:] @ a1[np.newaxis,:]
        delta_2 = delta_2 + d3.T @ a2[np.newaxis,:]
        
    delta_1 /= m
    delta_2 /= m
    
    delta_1[:,1:] = delta_1[:,1:] + theta_1[:,1:] * lmbda / m
    delta_2[:,1:] = delta_2[:,1:] + theta_2[:,1:] * lmbda / m
        
    return np.hstack((delta_1.ravel(order='F'), delta_2.ravel(order='F')))

def cost(params, L_input_size, HL_output_size, classes, X, y, lmbda):

    # initial thetas
    theta_1 = np.reshape(params[:HL_output_size * (L_input_size + 1)], (HL_output_size, L_input_size + 1), 'F')
    theta_2 = np.reshape(params[HL_output_size * (L_input_size + 1):], (classes, HL_output_size + 1), 'F')

    m = len(y)

    ones = np.ones((m,1))
    a1 = np.hstack((ones, X))
    a2 = sigmoid(a1 @ theta_1.T)
    a2 = np.hstack((ones, a2))
    h = sigmoid(a2 @ theta_2.T)
    
    # hot encode y values
    y_hot_encoded = pd.get_dummies(y.flatten())
    
    temp1 = np.multiply(y_hot_encoded, np.log(h))
    temp2 = np.multiply(1 - y_hot_encoded, np.log(1 - h))
    temp3 = np.sum(temp1 + temp2)
    
    sum1 = np.sum(np.sum(np.power(theta_1[:,1:],2), axis = 1))
    sum2 = np.sum(np.sum(np.power(theta_2[:,1:],2), axis = 1))
    
    return np.sum(temp3 / (-m)) + (sum1 + sum2) * lmbda / (2*m)

def checkBackpropagation(params, backpropagation_params, L_input_size, HL_output_size, classes, X, y, lmbda=0.0, eps = 0.0001):         

    for i in range(5):
        x = int(np.random.rand() * len(params))
        eps_vector = np.zeros((len(params), 1))
        eps_vector[x] = eps

        cost_high = cost(params + eps_vector.flatten(), L_input_size, HL_output_size, classes, X, y, lmbda)
        cost_low  = cost(params - eps_vector.flatten(), L_input_size, HL_output_size, classes, X, y, lmbda)

        gradient = (cost_high - cost_low) / float(2 * eps)

        print("Item: {0} - Real = {1:.6f} - Backpropagation = {2:.6f}".format(x, gradient, backpropagation_params[x]))

def analize(theta_1, theta_2, X, y):
    m = len(y)
    ones = np.ones((m,1))
    a1 = np.hstack((ones, X))
    a2 = sigmoid(a1 @ theta_1.T)
    a2 = np.hstack((ones, a2))
    h = sigmoid(a2 @ theta_2.T)
    return np.argmax(h, axis = 1) + 1
