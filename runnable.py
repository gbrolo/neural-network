import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from nn import *
from config import *
from utils import * 

# loading data
class_names, class_ids, data_path, train_number = train_data()
X, y = load_train_images(class_names, class_ids, data_path, train_number)
# print('X, y before: ', X[0], y[0])
# plt.imshow(X[0].reshape((28, 28), order = 'F'))
# plt.show()
X, y = shuffle_dataset(X, y)
# print('X, y after shuffle: ', X[0], y[0])
# plt.imshow(X[0].reshape((28, 28), order = 'F'))
# plt.show()
X_train, X_cross, X_test, y_train, y_cross, y_test = partition_dataset(X, y, train_number)
print('Training shapes X_train, y_train: ', X_train.shape, y_train.shape)
print('Cross shapes X_cross, y_cross: ', X_cross.shape, y_cross.shape)
print('Test shapes X_test, y_test: ', X_test.shape, y_test.shape)

# print('X, y train: ', X_train[0], y_train[0])
# plt.imshow(X_train[0].reshape((28, 28), order = 'F'))
# plt.show()

# print('X, y cross: ', X_cross[0], y_cross[0])
# plt.imshow(X_cross[0].reshape((28, 28), order = 'F'))
# plt.show()

# print('X, y test: ', X_test[0], y_test[0])
# plt.imshow(X_test[0].reshape((28, 28), order = 'F'))
# plt.show()

# show data
_, x = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
    for j in range(10):
        x[i,j].imshow(X[np.random.randint(X.shape[0])].reshape((28,28), order = 'F'))          
        x[i,j].axis('off')

plt.show()

# nn hyperparameters
# theta_1 has weights from input layer to hidden layer
# theta_2 has weights from hidden layer to output layer
L_input_size, HL_output_size, classes, lmbda = nn_hyperparameters()
theta_1 = initializeWeights(L_input_size, HL_output_size)
theta_2 = initializeWeights(HL_output_size, classes)

initial_params = np.hstack((theta_1.ravel(order='F'), theta_2.ravel(order='F')))
backpropagation_params = backpropagation(initial_params, L_input_size, HL_output_size, classes, X_train, y_train, lmbda)

# check if bp is made correctly
checkGradient(initial_params, backpropagation_params, L_input_size, HL_output_size, classes ,X_train, y_train, lmbda)

# optimize theta to find min
theta_opt = opt.fmin_cg(maxiter = 50, f = cost, x0 = initial_params, fprime = backpropagation, args = (L_input_size, HL_output_size, classes, X_train, y_train.flatten(), lmbda))
# theta_opt = gradient_descent(L_input_size, HL_output_size, classes, X, y.flatten(), initial_params, lmbda, iterations = 250)
print('theta_opt: ', theta_opt)

theta1_opt = np.reshape(theta_opt[:HL_output_size * (L_input_size + 1)], (HL_output_size, L_input_size + 1), 'F')
theta2_opt = np.reshape(theta_opt[HL_output_size * (L_input_size + 1):], (classes, HL_output_size + 1), 'F')

print('theta1_opt: ', theta1_opt)
print('theta2_opt: ', theta2_opt)

np.save('theta1_opt.npy', theta1_opt)
np.save('theta2_opt.npy', theta2_opt)

pred = analize(theta1_opt, theta2_opt, X_train, y_train)
percentage = np.mean(pred == y_train.flatten()) * 100

print('pred: ',pred)
print('percentage: ', percentage)