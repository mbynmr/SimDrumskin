from sim import sim
from analysis import basic, slice


# todo plans:
#  make the physics verlet
#  make the convolution for math with neighbouring coordinates


def main():
    t_end = 1e-2
    dt = 1e-6

    element_width = 6e-5
    total_width = 6e-3

    dims = [element_width, total_width]
    sim(dt, t_end, dims)
    # basic()
    slice(t_end, dims)


if __name__ == '__main__':
    main()
