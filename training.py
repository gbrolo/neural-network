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
X, y = shuffle_dataset(X, y)
X_train, X_cross, X_test, y_train, y_cross, y_test = partition_dataset(X, y, train_number)

print('Training shapes X_train, y_train: ', X_train.shape, y_train.shape)
print('Cross shapes X_cross, y_cross: ', X_cross.shape, y_cross.shape)
print('Test shapes X_test, y_test: ', X_test.shape, y_test.shape)

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

# check if bp is made correctly
backpropagation_params = backpropagation(initial_params, L_input_size, HL_output_size, classes, X_train, y_train, lmbda)
checkGradient(initial_params, backpropagation_params, L_input_size, HL_output_size, classes ,X_train, y_train, lmbda)

# optimize theta to find min
theta_opt = gradient_descent(L_input_size, HL_output_size, classes, X, y.flatten(), initial_params, lmbda)
theta1_opt = np.reshape(theta_opt[:HL_output_size * (L_input_size + 1)], (HL_output_size, L_input_size + 1), 'F')
theta2_opt = np.reshape(theta_opt[HL_output_size * (L_input_size + 1):], (classes, HL_output_size + 1), 'F')

# save thetas
np.save('theta1_opt.npy', theta1_opt)
np.save('theta2_opt.npy', theta2_opt)

percentage_train = get_theta_percentage(theta1_opt, theta2_opt, X_train, y_train)
percentage_cross = get_theta_percentage(theta1_opt, theta2_opt, X_cross, y_cross)
percentage_test = get_theta_percentage(theta1_opt, theta2_opt, X_test, y_test)

print('percentage_train: ', percentage_train)
print('percentage_cross: ', percentage_cross)
print('percentage_test: ', percentage_test)