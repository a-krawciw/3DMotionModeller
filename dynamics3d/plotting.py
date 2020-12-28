from dynamics3d.simulation import MotionStore3D
import matplotlib.pyplot as plt


def trace_plot_2d(body: MotionStore3D, ax1: str, ax2: str):
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(body.__getattribute__(ax1), body.__getattribute__(ax2))
    ax.set_xlabel(ax1.capitalize())
    ax.set_ylabel(ax2.capitalize())


def trace_plot_3d(body: MotionStore3D):
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.plot(body.x, body.y, body.z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot_against_time(body: MotionStore3D, **kwargs):
    f = plt.figure()
    ax = f.add_subplot(111)

    def plot_axis(axis: str):
        if not kwargs.get(axis, True):
            pass
        else:
            ax.plot(body.time, body.__getattribute__(axis),
                    label=f"{kwargs.get(axis + '_legend', 'Direction')} " + axis)

    plot_axis('x')
    plot_axis('y')
    plot_axis('z')
    ax.legend()
    ax.set_ylabel(kwargs.get("y_label", "Values"))
    ax.set_xlabel("Time (s)")
