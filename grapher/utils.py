import numpy as np
from matplotlib import colors

def random_color():
    return colors.hsv_to_rgb((np.random.rand(), np.random.rand(), 1))