import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


mpl.rcParams['lines.linewidth'] = 2.0


def three_frame(world, n_seq, seed=17):
    """
    Draw three timesteps.
    world: object with step, loop, and draw
    n_seq: 3-tuple, number of steps before each draw
    seed: random see for NumPy
    """
    np.random.seed(seed)
    plt.figure(figsize=(10, 4))

    for i, n in enumerate(n_seq):
        plt.subplot(1, 3, i + 1)
        world.loop(n)
        world.draw()

    plt.tight_layout()


def savefig(filename, **options):
    print("Saving figure to file", filename)
    plt.savefig(filename, **options)


def underride(d, **options):
    """
    Add key-value pairs to d only if key is not in d.
    d: dictionary
    options: keyword args to add to d
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d


def decorate(**options):
    """Decorate the current axes."""
    loc = options.pop("loc", "best")
    if options.pop("legend", True):
        legend(loc=loc)

    plt.gca().set(**options)
    plt.tight_layout()


def legend(**options):
    """Draws a legend only if there is at least one labeled item."""
    underride(options, loc="best", frameon=False)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **options)


def set_palette(*args, **kwds):
    """
    Set the matplotlib color cycler.
    returns: list of colors
    """
    reverse = kwds.pop('reverse', False)
    palette = sns.color_palette(*args, **kwds)

    palette = list(palette)
    if reverse:
        palette.reverse()

    cycler = plt.cycler(color=palette)
    plt.gca().set_prop_cycle(cycler)
    return palette
