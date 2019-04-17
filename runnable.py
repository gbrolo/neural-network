import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from utils import load_train_images 
from config import *
from nn import *

# loading data
class_names, class_ids, data_path, train_number = train_data()

X, y = load_train_images(class_names, class_ids, data_path, train_number)
# print(X.shape)
# print(X[0])
# print(y[0])

# show data
_, x = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
    for j in range(10):
        x[i,j].imshow(X[np.random.randint(X.shape[0])].reshape((50,50), order = 'F'))          
        x[i,j].axis('off')

plt.show()

# plt.figure(figsize=(10,10))
# columns = 5

# for i in range(columns):
#     plt.subplot(5 / columns + 1, columns, i + 1)
#     plt.imshow(X[i])

# plt.show()

# nn hyperparameters
L_input_size, HL_output_size, classes, lmbda = nn_hyperparameters()

# theta_1 has weights from input layer to hidden layer
theta_1 = initializeWeights(L_input_size, HL_output_size)

# theta_2 has weights from hidden layer to output layer
theta_2 = initializeWeights(HL_output_size, classes)

initial_params = np.hstack((theta_1.ravel(order='F'), theta_2.ravel(order='F')))
backpropagation_params = backpropagation(initial_params, L_input_size, HL_output_size, classes, X, y, lmbda)

# check if bp is made correctly
checkBackpropagation(initial_params, backpropagation_params, L_input_size, HL_output_size, classes ,X, y, lmbda)

# optimize theta to find min
theta_opt = opt.fmin_cg(maxiter = 50, f = cost, x0 = initial_params, fprime = backpropagation, args = (L_input_size, HL_output_size, classes, X, y.flatten(), lmbda))
print('theta_opt: ', theta_opt)

theta1_opt = np.reshape(theta_opt[:HL_output_size * (L_input_size + 1)], (HL_output_size, L_input_size + 1), 'F')
theta2_opt = np.reshape(theta_opt[HL_output_size * (L_input_size + 1):], (classes, HL_output_size + 1), 'F')

print('theta1_opt: ', theta1_opt)
print('theta2_opt: ', theta2_opt)

pred = analize(theta1_opt, theta2_opt, X, y)
percentage = np.mean(pred == y.flatten()) * 100

print('pred: ',pred)
print('percentage: ', percentage)