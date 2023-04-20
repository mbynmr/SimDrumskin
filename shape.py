import numpy as np


def retrieve_membrane_shape(size=None, shape='circle'):
    if size is None:
        size = [100, 100]
    X, Y = np.mgrid[:size[0], :size[1]]
    match shape:
        case 'circle':
            a = np.where((X / size[0] - 1 / 2) ** 2 + (Y / size[1] - 1 / 2) ** 2 <= 1 / 4, 1, 0)
        case 'square':
            a = np.ones(size)
        case 'circle_spikes':
            a = np.where((X / size[0] - 1 / 2) ** 2 + (Y / size[1] - 1 / 2) ** 2 <= 1 / 4, 1, 0)
        case _:
            raise ValueError()
    return a
