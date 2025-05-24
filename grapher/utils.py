import numpy as np
from matplotlib import colors
import itertools

def gen_palette(pal_name, n):
    match pal_name:
        case "grey":
            hues = np.full(n, 0)
            sats = np.full(n, 0)
            vals = np.linspace(0.3, 1, n)
        case "heat":
            hues_1 = np.full(n // 2, 0.7)
            hues_2 = np.full(n // 2, 1)
            hues = np.append(hues_1, hues_2)
            if n % 2 != 0:
                hues = np.insert(hues, n // 2 + 1, 1)
            sats = np.arange(n)
            sats = 2 * np.abs(sats - (n-1) / 2) / (n-1)
            sats = np.clip(sats, 0, 1)
            print(sats)
            vals = np.full(n, 1)
        case "random":
            hues = np.random.rand(n)
            sats = np.random.rand(n)
            vals = np.random.rand(n)
        case _:
            raise ValueError
    cols = map(lambda t: colors.hsv_to_rgb(t), zip(hues, sats, vals))
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