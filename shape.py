import numpy as np


def retrieve_membrane_shape(size=None, shape='circle'):
    if size is None:
        size = [100, 100]
    size = np.asarray(size) - 2
    X, Y = np.mgrid[:size[0], :size[1]]
    Xn = (X / size[0] - 1 / 2)
    Yn = (Y / size[1] - 1 / 2)
    match shape:
        case 'circle':
            a = np.where(Xn ** 2 + Yn ** 2 <= 1 / 4, 1, 0)
        case 'square':
            a = np.ones(size)
        case 'triangle':
            alpha = np.sqrt(0.333 * 4 * 3 ** (-0.5))  # todo make bigger or smaller!
            # a = np.zeros(size, dtype=np.bool_)
            c1 = np.where(-2 * Yn < alpha * 3 ** (-0.5), 1, 0)
            c2 = np.where(Yn - Xn * 3 ** 0.5 < alpha * 3 ** (-0.5), 1, 0)
            c3 = np.where(Yn + Xn * 3 ** 0.5 < alpha * 3 ** (-0.5), 1, 0)
            a = np.where(c1 + c2 + c3 == 3, 1, 0)
        case 'circle_spikes':
            a = np.where(Xn ** 2 + Yn ** 2 <= 1 / 4, 1, 0)
        case _:
            raise ValueError()
    return np.pad(a, 1, constant_values=0)
