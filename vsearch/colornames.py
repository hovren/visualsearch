import os

import numpy as np

COLOR_NAMES = ['black', 'blue', 'brown', 'grey', 'green', 'orange',
               'pink', 'purple', 'red', 'white', 'yellow']
COLOR_RGB = [[0, 0, 0] , [0, 0, 1], [.5, .4, .25] , [.5, .5, .5] , [0, 1, 0] , [1, .8, 0] ,
             [1, .5, 1] ,[1, 0, 1], [1, 0, 0], [1, 1, 1 ] , [ 1, 1, 0 ]]


def colornames_image(im, w2c, mode='index'):
    im = im.astype('double')
    idx = np.floor(im[..., 0] / 8) + 32 * np.floor(im[..., 1] / 8) + 32 * 32 * np.floor(im[..., 2] / 8)
    m = w2c[idx.astype('int')]

    if mode == 'index':
        return np.argmax(m, 2)
    elif mode == 'probability':
        return m
    else:
        raise ValueError("No such mode: '{}'".format(mode))

def cname_file_for_image(path):
    root, _ = os.path.splitext(path)
    cname_file = os.path.join(root + '.cname.h5')
    return cname_file

def calculate_colornames(image, roi=None):
    raise NotImplementedError("Nope, not done yet")
