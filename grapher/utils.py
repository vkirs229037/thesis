import numpy as np
from matplotlib import colors
import itertools

def gen_palette():
    hues = np.linspace(0, 1, 24)
    sats = np.linspace(0.75, 1, 2)
    vals = np.linspace(0.75, 1, 2)
    cols = sorted(map(lambda t: colors.hsv_to_rgb(t), itertools.product(hues, sats, vals)), key=lambda t: t[1])
    return list(cols)