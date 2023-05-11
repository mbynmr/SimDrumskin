from sim import sim, sim_dem
from analysis import basic, slice_viewer


# todo plans:
#  make the physics verlet


def main():
    t_end = 2e-3
    dt = 1e-7

    upscale_factor = 1e3  # python doesn't enjoy such small numbers so increase the size of everything!

    element_width = 6e-5 * upscale_factor
    total_width = 6e-3 * upscale_factor
    # create a lock button where either one is locked. if you change one smoothly the other compensates (jumping near
    # to its actual value to get integer values)

    b = 0.99  # damping coefficient
    b = 0
    c = 1e3 * upscale_factor  # speed
    # (0, 1) is at 1.31e4 for b=0
    freq = 1.305e4 * 2.3  # driving frequency
    amp = 1e-5 * upscale_factor  # amplitude
    k = 1e9 * upscale_factor
    m = total_width ** 2 * (100e-6 * upscale_factor) * 1 * upscale_factor  # roughly correct expected mass of PDMS film
    g = 9.81 * upscale_factor  # * 1000

    dm = (element_width / total_width) ** 2 * m

    # sim(dt, t_end, element_width, total_width, freq, amp, b, c)
    sim_dem(dt, t_end, element_width, total_width, freq, amp, b, k=k, dm=dm, g=g)
    basic()
    slice_viewer(t_end, element_width, total_width)


if __name__ == '__main__':
    main()
