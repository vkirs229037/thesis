from graph import Graph, GraphKind, Vertex
from typing import Tuple, List
import numpy as np
import random

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

# Является ли граф эйлеровым
def is_euler(g: Graph) -> bool:
    for v in g.vertices:
        degree = np.sum(np.where(g[v.id, :] != 0, 1, 0))
        if g.kind == GraphKind.Directed:
            degree += np.sum(np.where(g[:, v.id] != 0, 1, 0))
        if degree % 2 != 0:
            return False
    return True

# Алгоритм Флёри
# Предполагается что уже прошла проверка на эйлеровость графа
def fleury(g: Graph) -> List[Tuple[int, int]]:
    if g.kind == GraphKind.Directed:
        return fleury_directed(g)
    else:
        return fleury_undirected(g)

def fleury_directed(g: Graph) -> List[Tuple[int, int]]:
    all_edges = list(g.edges.keys())
    print(g.edges)
    marked = {}
    result = []
    first_edge = random.choice(all_edges)
    marked[first_edge] = True
    result.append(first_edge)
    while set(marked.keys()) != set(all_edges):
        e = result[-1]
        v = e[1]
        print("------------------------")
        print(v)
        print(marked)
        conns = np.nonzero(g[v, :])[0]
        edges = set([(v, w) for w in conns if marked.get((v, w)) != True])
        print(edges)
        bridges = set()
        for w in conns:
            print(w, marked.get((v, w)))
            if v == w:
                continue
            c = g.edges[(v, w)]
            del g.edges[(v, w)]
            g[v, w] = 0
            r_m = reachability_matrix(g)
            if r_m[v, w] == 0:
                # Нашли мост
                g.edges[(v, w)] = c
                g[v, w] = c
                bridges.add((v, w))
        non_bridges = edges - bridges
        if len(non_bridges) == 0:
            result.append(list(bridges)[0])
            marked[list(bridges)[0]] = True
        else:
            result.append(list(non_bridges)[0])
            marked[list(non_bridges)[0]] = True
    return result

def fleury_undirected(g: Graph) -> List[Tuple[int, int]]:
    all_edges = list(g.edges.keys())
    print(g.edges)
    marked = {}
    result = []
    first_edge = random.choice(all_edges)
    marked[first_edge] = True
    marked[(first_edge[1], first_edge[0])] = True
    result.append(first_edge)
    while set(marked.keys()) != set(all_edges):
        e = result[-1]
        v = e[1]
        print("------------------------")
        print(v)
        print(marked)
        conns = np.nonzero(g[v, :])[0]
        edges = set([(v, w) for w in conns if marked.get((v, w)) != True or marked.get((w, v)) != True])
        print(edges)
        bridges = set()
        for w in conns:
            print(w, marked.get((v, w)))
            if v == w:
                continue
            c = g[v, w]
            g[v, w] = 0
            g[w, v] = 0
            r_m = reachability_matrix(g)
            if r_m[v, w] == 0:
                # Нашли мост
                bridges.add((v, w))
            g[v, w] = c
            g[w, v] = c
        non_bridges = edges - bridges
        if len(non_bridges) == 0:
            new_e = list(bridges)[0]
        else:
            new_e = list(non_bridges)[0]
        result.append(new_e)
        marked[new_e] = True
        marked[new_e[1], new_e[0]] = True
    return result


# Задача китайского почтальона
def chinesepostman(g: Graph) -> List[int]:
    raise NotImplementedError