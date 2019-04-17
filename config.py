
def train_data():
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

    return class_names, class_ids, data_path, train_number

def nn_hyperparameters():
    L_input_size = 50 * 50
    HL_output_size = 75
    classes = 10
    lmbda = 0.0001

    return L_input_size, HL_output_size, classes, lmbda