import numpy as np
from tqdm import tqdm
from scipy.signal import convolve2d

from shape import retrieve_membrane_shape


def sim(dt, t_end, element_width, total_width, freq=2e3, amp=0.1, b=0, c=1e3):
    # todo sort units everywhere, make sure it's all just in metres and seconds
    grid = [int(total_width / element_width), int(total_width / element_width)]
    # grid = [200, 200]
    shape = retrieve_membrane_shape(size=grid, shape='circle')

    # dt = 0.01  # todo make dt from maximum acceleration?
    # dt = 0.01 / (amp * freq ** 2)
    times = np.arange(int(t_end / dt)) * dt

    # units
    if b == 0:
        damping = False
    else:
        damping = True

    # constants
    dx = 1 / (grid[1] - 2)
    k0 = (c * dt / dx) ** 2
    k1 = (1 + (b * dt / 2))
    c0 = (2 - 4 * k0) / k1
    c1 = (1 - (b * dt / 2)) / k1
    c2 = k0 / k1
    kernel = np.array([[0, c2, 0], [c2, c0, c2], [0, c2, 0]])
    kernel_damping = np.array([[0, k0, 0], [k0, 2 - 4 * k0, k0], [0, k0, 0]])

    # get indexes of membrane and support
    xm, ym = np.nonzero(shape)
    xs, ys = np.nonzero(1 - shape)

    M = np.zeros((*grid, 3))

    # predefine support positions to save calculation time, though this will be slower if the simulation is very long
    support = amp * np.sin(2 * np.pi * freq * times)
    M[xs, ys, 0] = amp * np.sin(2 * np.pi * freq * (0 - 2 * dt))
    M[xs, ys, 1] = amp * np.sin(2 * np.pi * freq * (0 - 1 * dt))

    with open('outputs/output.txt', 'w') as outputfile, open('outputs/outputbig.txt', 'w') as outputbigfile:
        for i, t in tqdm(enumerate(times), total=len(times)):
            M[xs, ys, 2] = support[i]

            # use finite difference formula to calculate new values of displacement of membrane
            if not damping:
                # no convolution: use for small arrays (less than 100,000 elements in first 2 dimensions)
                # M[xm, ym, 2] = c0 * M[xm, ym, 1] - c1 * M[xm, ym, 0] + c2 * (
                #         M[xm + 1, ym, 1] + M[xm, ym + 1, 1] + M[xm - 1, ym, 1] + M[xm, ym - 1, 1]
                # )

                # convolution: use for large arrays (more than 100,000 elements in first 2 dimensions)
                M[xm, ym, 2] = convolve2d(M[..., 1], kernel, mode='same').reshape(M.shape[0:2])[xm, ym] - \
                               c1 * M[xm, ym, 0]
                # manually (not working?)
                # M_1F = np.fft.fft2(M[..., 1])
                # M_1F[:kernel.shape[0], :kernel.shape[1]] = kernel_1 + M_1F[:kernel.shape[0], :kernel.shape[1]]
                # M[xm, ym, 2] = np.abs(np.fft.ifft2(M_1F)).reshape(M.shape[0:2])[xm, ym] - c1 * M[xm, ym, 0]

            else:
                # no convolution: use most of the time
                M[xm, ym, 2] = 2 * M[xm, ym, 1] - M[xm, ym, 0] + k0 * (
                        M[xm + 1, ym, 1] + M[xm, ym + 1, 1] + M[xm - 1, ym, 1] + M[xm, ym - 1, 1] - 4 * M[xm, ym, 1]
                ) - b * (M[xm, ym, 1] - M[xm, ym, 0]) * (M[xm, ym, 1] - M[xm, ym, 0])

                # convolution: use for very large arrays (more than 1,000,000 elements in first 2 dimensions)
                # M[xm, ym, 2] = convolve2d(M[..., 1], kernel_damping, mode='same').reshape(M.shape[0:2])[xm, ym] - (
                #         M[xm, ym, 0] + b * (M[xm, ym, 1] - M[xm, ym, 0]) * (M[xm, ym, 1] - M[xm, ym, 0])
                # )

            # recording data
            if i % 10 == 0:  # only write every ___ timestep
                outputfile.writelines(f"{i} {t:.6g} {np.sqrt(np.mean(np.square(M[xm, ym, 2]))):.6g} {support[i]:.6g}\n")
                # write step, time, rms pos of membrane, pos of support
            if i % int((t_end / dt) / 500) == 0:  # write 500 total frames
                for n in range(M.shape[1]):
                    outputbigfile.writelines(f"{M[int(grid[0] / 2), n, 2]:.3g} ")
                    # write position of every element across the middle section
                outputbigfile.writelines(f"\n")

            # roll on to the next timestep
            M[..., 0] = M[..., 1]
            M[..., 1] = M[..., 2]
            # np.roll(M, shift=1, axis=2)  # any way to get the roll to work? It would be faster

def sim_dem(dt, t_end, element_width, total_width, freq=2e3, amp=1e-5, b=0, k=1e6, dm=1e-9):
    # todo sort units everywhere, make sure it's all just in metres and seconds
    grid = [int(total_width / element_width), int(total_width / element_width)]
    # grid = [200, 200]
    shape = retrieve_membrane_shape(size=grid, shape='circle')

    # dt = 0.01  # todo make dt from maximum acceleration?
    # dt = 0.01 / (amp * freq ** 2)
    times = np.arange(int(t_end / dt)) * dt

    # units
    if b == 0:
        damping = False
    else:
        damping = True

    # constants
    dx = element_width
    dx2 = dx ** 2
    km = - k / dm
    dt2 = dt ** 2

    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    # get indexes of membrane and support
    xm, ym = np.nonzero(shape)
    xs, ys = np.nonzero(1 - shape)

    M = np.zeros((*grid, 3))

    # predefine support positions to save calculation time, though this will be slower if the simulation is very long
    support = amp * np.sin(2 * np.pi * freq * times)
    M[xs, ys, 0] = amp * np.sin(2 * np.pi * freq * (0 - 2 * dt))
    M[xs, ys, 1] = amp * np.sin(2 * np.pi * freq * (0 - 1 * dt))

    with open('outputs/output.txt', 'w') as outputfile, open('outputs/outputbig.txt', 'w') as outputbigfile:
        for i, t in tqdm(enumerate(times), total=len(times)):
            M[xs, ys, 2] = support[i]

            # use finite difference formula to calculate new values of displacement of membrane
            if not damping:
                # no convolution: use for small arrays (less than 100,000 elements in first 2 dimensions)
                # prev_dz = M[xm, ym, 1] - M[xm, ym, 0]  # = speed * timestep
                # f = convolve2d(M[..., 1], kernel, mode='same')
                # f = convolve2d(f, kernel, mode='same')
                f1 = (np.sqrt((M[xm, ym, 1] - M[xm + 1, ym, 1]) ** 2 + dx2) - dx) * 2 * ((M[xm, ym, 1] - M[xm + 1, ym, 1] >= 0) - 0.5)
                f2 = (np.sqrt((M[xm, ym, 1] - M[xm, ym + 1, 1]) ** 2 + dx2) - dx) * 2 * ((M[xm, ym, 1] - M[xm, ym + 1, 1] >= 0) - 0.5)
                f3 = (np.sqrt((M[xm, ym, 1] - M[xm - 1, ym, 1]) ** 2 + dx2) - dx) * 2 * ((M[xm, ym, 1] - M[xm - 1, ym, 1] >= 0) - 0.5)
                f4 = (np.sqrt((M[xm, ym, 1] - M[xm, ym - 1, 1]) ** 2 + dx2) - dx) * 2 * ((M[xm, ym, 1] - M[xm, ym - 1, 1] >= 0) - 0.5)
                M[xm, ym, 2] = (f1 + f2 + f3 + f4) * km * dt2 + 2 * M[xm, ym, 1] - M[xm, ym, 0]
            else:
                print("not yet got damping")
            # recording data
            if i % 10 == 0:  # only write every ___ timestep
                a = M[xm, ym, 2]
                a = np.where(np.inf == np.where(a == np.NaN, 0, a), np.nanmax(a), a)
                outputfile.writelines(f"{i} {t:.6g} {np.sqrt(np.mean(np.square(a))):.6g} {support[i]:.6g}\n")
                # write step, time, rms pos of membrane, pos of support
            if i % int((t_end / dt) / 250) == 0:  # write 500 total frames
                for n in range(M.shape[1]):
                    a = M[int(grid[0] / 2), n, 2]
                    a = np.where(np.inf == np.where(a == np.NaN, 0, a), np.nanmax(a), a)
                    outputbigfile.writelines(f"{a:.3g} ")
                    # write position of every element across the middle section
                outputbigfile.writelines(f"\n")

            # roll on to the next timestep
            M[..., 0] = M[..., 1]
            M[..., 1] = M[..., 2]
            # np.roll(M, shift=1, axis=2)  # any way to get the roll to work? It would be faster
