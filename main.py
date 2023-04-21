from sim import sim
from analysis import basic, slice


# todo plans:
#  make the physics verlet
#  make the convolution for math with neighbouring coordinates


def main():
    dt = 0.01
    t_end = 50
    sim(dt, t_end)
    # basic()
    slice(t_end)


if __name__ == '__main__':
    main()
