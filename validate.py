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

# loading thetas
theta1_opt = np.load('theta1_opt_69-05.npy')
theta2_opt = np.load('theta2_opt_69-05.npy')

percentage_train = get_theta_percentage(theta1_opt, theta2_opt, X_train, y_train)
percentage_cross = get_theta_percentage(theta1_opt, theta2_opt, X_cross, y_cross)
percentage_test = get_theta_percentage(theta1_opt, theta2_opt, X_test, y_test)

print('percentage_train: ', percentage_train)
print('percentage_cross: ', percentage_cross)
print('percentage_test: ', percentage_test)