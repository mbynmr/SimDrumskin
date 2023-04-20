from sim import sim
from analysis import basic, slice


def main():
    dt = 0.01
    t_end = 100
    sim(dt, t_end)
    # basic()
    slice(t_end)


if __name__ == '__main__':
    main()
