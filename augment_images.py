import sys
import Augmentor

def augment(folder):    
    p = Augmentor.Pipeline(source_directory = folder, save_format = "png")
    p.flip_left_right(0.5)
    p.black_and_white(0.1)
    p.gaussian_distortion(probability = 0.4, grid_width = 7, grid_height = 6, magnitude = 6, corner = "ul", method = "in", mex = 0.5, mey = 0.5, sdx = 0.05, sdy = 0.05)

    p.rotate(0.3, 10,10)
    p.skew(0.4,0.5)
    p.skew_tilt(0.6,0.8)
    p.skew_left_right(0.5, magnitude = 0.8)
    p.sample(4000)

def augment_images():
    folders = ['downloads/circle', 'downloads/egg', 'downloads/happy', 'downloads/house', 'downloads/mickey', 'downloads/question', 'downloads/sad', 'downloads/square', 'downloads/tree', 'downloads/triangle']

    for folder in folders:
        print('Augmenting' + folder)
        augment(folder)

    print('Finished augmenting')

augment_images()