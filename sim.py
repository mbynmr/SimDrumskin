import numpy as np
from tqdm import tqdm

from shape import retrieve_membrane_shape


def sim(dt, t_end):
    size = [100, 100]
    shape = retrieve_membrane_shape(size=[size[0] - 2, size[1] - 2], shape='circle')
    shape = np.pad(shape, 1, constant_values=0)

    freq = float(1)
    amp = float(0.1)
    # dt = 0.01  # todo make dt from maximum acceleration?
    # dt = 0.01 / (amp * freq ** 2)
    times = np.arange(int(t_end / dt)) * dt
    print(times[0])
    print(times[1])
    print(times[-1])

    # todo units
    a = 1
    b = 0  # damping coefficient
    c = 0.2  # speed

    # constants
    dx = 1 / (size[1] - 2)
    k0 = (c * dt / dx) ** 2
    k1 = (1 + (b * dt / 2))
    c0 = (2 - 4 * k0) / k1
    c1 = (1 - (b * dt / 2)) / k1
    c2 = k0 / k1

    # get indexes of membrane and support
    xm, ym = np.nonzero(shape)
    xs, ys = np.nonzero(1 - shape)

    M = np.zeros((*size, 3))

    # predefine support positions to save calculation time, though this will be slower if the simulation is very long
    support = amp * np.sin(2 * np.pi * freq * c * times / a)
    M[xs, ys, 0] = amp * np.sin(2 * np.pi * freq * c * (0 - 2 * dt) / a)
    M[xs, ys, 1] = amp * np.sin(2 * np.pi * freq * c * (0 - 1 * dt) / a)

    with open('outputs/output.txt', 'w') as outputfile, open('outputs/outputbig.txt', 'w') as outputbigfile:
        for i, t in tqdm(enumerate(times), total=len(times)):
            M[xs, ys, 2] = support[i]

            # use finite difference formula to calculate new values of displacement of membrane
            M[xm, ym, 2] = c0 * M[xm, ym, 1] - c1 * M[xm, ym, 0] + c2 * (
                    M[xm + 1, ym, 1] + M[xm, ym + 1, 1] + M[xm - 1, ym, 1] + M[xm, ym - 1, 1]
            )  # todo place for a convolution: math with neighbouring coordinates is it's bread and butter
            if i % 10 == 0:  # only write every ___ timestep
                outputfile.writelines(f"{i} {t:.6g} {np.sqrt(np.mean(np.square(M[xm, ym, 2]))):.6g} {support[i]:.6g}\n")
                # write step, time, rms pos of membrane, pos of support
            if i % int((t_end / dt) / 1000) == 0:
                for n in range(M.shape[1]):
                    outputbigfile.writelines(f"{M[int(size[0] / 2), n, 2]:.6g} ")
                    # write position of every element across the middle section
                outputbigfile.writelines(f"\n")

            # roll on to the next timestep
            M[..., 0] = M[..., 1]
            M[..., 1] = M[..., 2]
            # np.roll(M, shift=1, axis=2)  # todo check it does what i expect, if so it should be faster!
