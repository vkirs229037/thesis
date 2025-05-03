from graph import Graph, GraphKind, Vertex
from typing import Tuple, List
import numpy as np

def connections(g: Graph, v: int) -> set[int]:
    n = g.n
    s = set()
    s.add(v)
    prev_s = set()
    while not (prev_s == s):
        prev_s = s
        temp_s = set()
        for xi in s:
            for i in range(n):
                if g[xi, i] != 0:
                    temp_s.add(i)
        s = s.union(temp_s)
    return s

def reachability_matrix(g: Graph) -> np.ndarray[int]:
    result = np.zeros((g.n, g.n), dtype=np.dtype(int))
    for v in g.vertices:
        conns = connections(g, v.id)
        result[v.id, list(conns)] = 1
    return result

# Алгоритм Дейкстры. Возвращается кратчайший путь и ID вершин, вошедших в этот путь
def dijkstra(g: Graph, s: int, t: int) -> Tuple[int, List[int]]:
    l_temp = {}
    l_const = {s: 0}
    for v in g.vertices:
        if v.id != s:
            l_temp[v.id] = np.iinfo(np.int64).max
    p = s
    while p != t:
        for xi in l_temp.keys():
            c = g[p, xi]
            if c == 0:
                continue
            l_temp[xi] = min(l_temp[xi], l_const[p] + c)
        min_l = min(l_temp.items(), key = lambda tup: tup[1])
        xi_star = min_l[0]
        l_const[xi_star] = min_l[1]
        del l_temp[xi_star]
        p = xi_star
    d = l_const[p]
    path = [p]
    while p != s:
        xi = p
        for xi_prime in l_const.keys():
            if xi_prime == p:
                break
            c = g[xi_prime, xi]
            if (l_const[xi_prime] + c == l_const[xi]):
                p = xi_prime
                path.append(xi_prime)
                break
    path.reverse()
    return (d, path)