import numpy as np
from matplotlib import colors
import itertools

def gen_palette():
    hues = np.linspace(0, 1, 24)
    sats = np.linspace(0.75, 1, 2)
    vals = np.linspace(0.75, 1, 2)
    cols = sorted(map(lambda t: colors.hsv_to_rgb(t), itertools.product(hues, sats, vals)), key=lambda t: t[1])
    return list(cols)

def gen_random_graph(n):
    s = "vertex {\n"
    vs = range(1, n+1)
    for i in vs:
        s += "\t" + str(i) + ";\n"
    s += "}\ngraph {\n\tundirected;\n"
    for i in vs:
        stmt = f"\t{i} > "
        l = np.random.rand() > 0.5
        if l:
            stmt += "["
            for j in vs:
                if np.random.rand() > 0.5:
                    stmt += str(j) + " "
            stmt += "]"
        else:
            stmt += str(np.random.randint(1, n+1))
        s += stmt + " #1;\n"
    s += "}"
    return s