from sim import sim, sim_dem
from analysis import basic, slice_viewer


# todo plans:
#  make the physics verlet


def main():
    t_end = 1e-3
    dt = 5e-8

    element_width = 6e-5 * 1e3
    total_width = 6e-3 * 1e3
    # create a lock button where either one is locked. if you change one smoothly the other compensates (jumping near
    # to its actual value to get integer values)

    b = 0  # damping coefficient
    c = 1e3  # speed
    freq = 2e3  # driving frequency
    amp = 1e-5 * 1e3  # amplitude
    k = 1e4 * 1e-3
    m = 5e-9  # roughly correct expected mass of PDMS film
    dm = (element_width / total_width) ** 2 * m

    # sim(dt, t_end, element_width, total_width, freq, amp, b, c)
    sim_dem(dt, t_end, element_width, total_width, freq, amp, b, k=k, dm=dm)
    # basic()
    slice_viewer(t_end, element_width, total_width)


if __name__ == '__main__':
    main()
