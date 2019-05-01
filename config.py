# Config for neural network classifier
# @author: gbrolo

# training data config
def train_data():
    class_names = [
        'circle', 
        'egg', 
        'happy', 
        'house', 
        'mickey', 
        'question', 
        'sad', 
        'square', 
        'tree', 
        'triangle'
        ]

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
    train_number = 5000

    return class_names, class_ids, data_path, train_number

# class ids dictionaries
def get_class_ids():
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

    ids_to_class = {
        '0': 'circle',
        '1': 'egg',
        '2': 'happy',
        '3': 'house',
        '4': 'mickey',
        '5': 'question',
        '6': 'sad',
        '7': 'square',
        '8': 'tree',
        '9': 'triangle'
    }

    return class_ids, ids_to_class

# hyperparameters for neural network
def nn_hyperparameters():
    L_input_size = 28 * 28      # input layer size
    HL_output_size = 400         # hidden layer size
    classes = 10                # number of classes
    lmbda = 3                 # learning rate

    return L_input_size, HL_output_size, classes, lmbda