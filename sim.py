import numpy as np
from tqdm import tqdm
from scipy.signal import convolve2d

from shape import retrieve_membrane_shape


def sim(dt, t_end, dims):
    element_width, total_width = dims
    grid = [int(total_width / element_width), int(total_width / element_width)]
    # grid = [200, 200]
    shape = retrieve_membrane_shape(size=grid, shape='circle')

    freq = 2.3e3
    amp = 0.1
    # dt = 0.01  # todo make dt from maximum acceleration?
    # dt = 0.01 / (amp * freq ** 2)
    times = np.arange(int(t_end / dt)) * dt

    # todo sort units everywhere, make sure it's all just in metres and seconds
    # units
    b = 0  # damping coefficient
    c = 1e3  # speed
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

            kernel_1 = np.array([[0, c2, 0], [c2, c0, c2], [0, c2, 0]])
            kernel_0 = np.array([[0, 0, 0], [0, -c1, 0], [0, 0, 0]])  # useless?

            # use finite difference formula to calculate new values of displacement of membrane
            if not damping:
                # no convolution
                # M[xm, ym, 2] = c0 * M[xm, ym, 1] - c1 * M[xm, ym, 0] + c2 * (
                #         M[xm + 1, ym, 1] + M[xm, ym + 1, 1] + M[xm - 1, ym, 1] + M[xm, ym - 1, 1]
                # )

                # efficient use of convolution
                M[xm, ym, 2] = convolve2d(M[..., 1], kernel_1, mode='same').reshape(M.shape[0:2])[xm, ym] -\
                               c1 * M[xm, ym, 0]
                # manually (not working?)
                # M_1F = np.fft.fft2(M[..., 1])
                # M_1F[:kernel_1.shape[0], :kernel_1.shape[1]] = kernel_1 + M_1F[:kernel_1.shape[0], :kernel_1.shape[1]]
                # M[xm, ym, 2] = np.abs(np.fft.ifft2(M_1F)).reshape(M.shape[0:2])[xm, ym] - c1 * M[xm, ym, 0]

                # full convolution
                # M[xm, ym, 2] = (convolve2d(M[..., 1], kernel_1, mode='same') +
                #                 convolve2d(M[..., 0], kernel_0, mode='same')).reshape(M.shape[0:2])[xm, ym]

            else:
                M[xm, ym, 2] = 2 * M[xm, ym, 1] - M[xm, ym, 0] + k0 * (
                        M[xm + 1, ym, 1] + M[xm, ym + 1, 1] + M[xm - 1, ym, 1] + M[xm, ym - 1, 1] - 4 * M[xm, ym, 1]) -\
                               b * (M[xm, ym, 1] - M[xm, ym, 0]) * (M[xm, ym, 1] - M[xm, ym, 0])

            # recording data
            if i % 10 == 0:  # only write every ___ timestep
                outputfile.writelines(f"{i} {t:.6g} {np.sqrt(np.mean(np.square(M[xm, ym, 2]))):.6g} {support[i]:.6g}\n")
                # write step, time, rms pos of membrane, pos of support
            if i % int((t_end / dt) / 1000) == 0:  # write 1000 frames
                for n in range(M.shape[1]):
                    outputbigfile.writelines(f"{M[int(grid[0] / 2), n, 2]:.3g} ")
                    # write position of every element across the middle section
                outputbigfile.writelines(f"\n")

            # roll on to the next timestep
            M[..., 0] = M[..., 1]
            M[..., 1] = M[..., 2]
            # np.roll(M, shift=1, axis=2)  # any way to get the roll to work? It would be faster
