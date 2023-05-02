import numpy as np
import matplotlib.pyplot as plt
import time


def basic():
    data = np.loadtxt('outputs/output.txt')
    i = data[:, 0]  # iteration number
    t = data[:, 1]  # time
    a = data[:, 2]  # rms amplitude of non-support
    s = data[:, 3]  # support
    plt.plot(t, a, 'b-')
    plt.plot(t, s, 'r-')
    plt.ylabel('rms displacement')
    plt.xlabel('time')
    plt.legend({'membrane', 'support'})
    plt.show()


def slice_viewer(t_end, element_width, total_width):
    # data = np.loadtxt('outputs/output.txt')
    slice = np.loadtxt('outputs/outputbig.txt')
    # i = data[:, 0]  # iteration number
    # t = data[:, 1]  # time
    # a = data[:, 2]  # rms amplitude of non-support
    # s = data[:, 3]  # support
    plt.ion()
    figure, ax = plt.subplots()
    plt.ylabel('vertical displacement')
    plt.xlabel('horizontal position')
    line1, = ax.plot(np.linspace(start=-total_width / 2, stop=total_width / 2, num=slice.shape[1]), slice[0, :])
    # ax.set_ylim(bottom=min(slice), top=max(slice))
    plt.ylim([np.amin(slice), np.amax(slice)])
    # plt.show
    t_start = time.time()
    for i, frame in enumerate(slice):
        # line1.set_xdata(x)
        line1.set_ydata(frame)
        # plt.ylim([np.amin(frame), np.amax(frame)])
        plt.title(f"t = {i * t_end / slice.shape[0]:.2e}s out of {t_end:.2e}s")
        figure.canvas.draw()
        figure.canvas.flush_events()
        sleep = i * (1 / 30) - (time.time() - t_start)  # how long to hold this frame for to get 30fps
        if sleep > 0:
            time.sleep(sleep)
    plt.close()
    plt.ioff()
