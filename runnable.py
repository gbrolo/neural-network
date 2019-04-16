import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from utils import load_train_images 

# loading data
class_names = ['circle', 'egg', 'happy', 'house', 'mickey', 'question', 'sad', 'square', 'tree', 'triangle']
class_ids = {
    'circle': 0,
    'egg': 1,
    'happy': 2,
    'house': 3,
    'mickey': 4,
    'question': 5,
    'sad': 6,
    'square': 7,
    'tree': 8,
    'triangle': 9
}
data_path = 'data'
train_number = 1000

X, y = load_train_images(class_names, class_ids, data_path, train_number)
print(X[0])
print(y[0])

plt.figure(figsize=(20,10))
columns = 5

for i in range(columns):
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.imshow(X[i])

plt.show()