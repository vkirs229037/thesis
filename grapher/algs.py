from graph import Graph, GraphKind, Vertex
from typing import Tuple, List, Dict, Set
import numpy as np
import random
import copy

###
### Задачи
###
def connections(g: Graph, v: int) -> set[int]:
    n = g.n
    s = set()
    s.add(v)
    prev_s = set()
    while not (prev_s == s):
        prev_s = s
        for xi in prev_s:
            conns = set(np.nonzero(g[xi, :])[0])
            s = s.union(conns)
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

# Алгоритм Флойда
def floyd(g: Graph) -> Tuple[np.ndarray, np.ndarray] | bool:
    c = copy.deepcopy(g[:, :])
    vs = np.array([v.id for v in g.vertices])
    theta = np.repeat(vs.reshape(-1, 1), g.n, axis=1)
    c[c == 0] = np.iinfo(np.int32).max
    k = 0
    while k != g.n:
        for i in range(g.n):
            if k == i or c[i, k] == np.inf:
                continue
            for j in range(g.n):
                if j == i or c[k, j] == np.inf: 
                    continue
                s = c[i, k] + c[k, j]
                cij = c[i, j]
                if s < cij:
                    c[i, j] = s
                    theta[i, j] = theta[k, j]
            if c[i, i] < 0:
                return False
        k += 1
    c[c >= np.iinfo(np.int32).max] = 0
    return c, theta


# Алгоритм Флёри
# Предполагается что уже прошла проверка на эйлеровость графа
def fleury(g: Graph) -> List[Tuple[int, int]]:
    if g.kind == GraphKind.Directed:
        return fleury_directed(g)
    else:
        return fleury_undirected(g)

def fleury_directed(g: Graph) -> List[Tuple[int, int]]:
    all_edges = set(g.edges.keys())
    marked = set()
    result = []
    first_edge = random.choice(list(all_edges))
    marked.add(first_edge)
    result.append(first_edge)
    while marked != all_edges:
        e = result[-1]
        v = e[1]
        conns = np.nonzero(g[v, :])[0]
        edges = set([(v, w) for w in conns if (v, w) not in marked])
        bridges = set()
        for w in conns:
            if v == w:
                continue
            c = g.edges[(v, w)]
            del g.edges[(v, w)]
            g[v, w] = 0
            r_m = reachability_matrix(g)
            if r_m[v, w] == 0 or r_m[w, v] == 0:
                # Нашли мост
                bridges.add((v, w))
            g.edges[(v, w)] = c
            g[v, w] = c
        non_bridges = edges - bridges
        if len(non_bridges) == 0:
            new_e = list(bridges)[0]
        else:
            new_e = list(non_bridges)[0]
        result.append(new_e)
        marked.add(new_e)
    return result

def fleury_undirected(g: Graph) -> List[Tuple[int, int]]:
    all_edges = set(g.edges.keys())
    marked = set()
    result = []
    first_edge = random.choice(list(all_edges))
    marked.add(first_edge)
    marked.add((first_edge[1], first_edge[0]))
    result.append(first_edge)
    while marked != all_edges:
        e = result[-1]
        v = e[1]
        conns = np.nonzero(g[v, :])[0]
        edges = set([(v, w) for w in conns if (v, w) not in marked or (w, v) not in marked])
        bridges = set()
        for w in conns:
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
        marked.add(new_e)
        marked.add((new_e[1], new_e[0]))
    return result

# Общий обход графа с запоминанием вершин
def walk(g: Graph, p0: int) -> Set[int]:
    visited = set([p0])
    S = [p0]
    while len(S) > 0:
        q = S.pop()
        succs = set(np.nonzero(g[q, :])[0])
        for p in succs:
            if p not in visited:
                visited.add(p)
                S.append(p)
    return visited

# Нахождение компонент связности
def conn_comps(g: Graph) -> List[Set[int]]:
    visited = set()
    comps = []
    for v in [v.id for v in g.vertices]:
        if v in visited:
            continue
        current_visited = walk(g, v)
        visited.update(current_visited)
        if len(comps) > 0 and comps[-1].intersection(current_visited) != set():
            comps[-1].update(current_visited)
        else:
            comps.append(set(current_visited))
    return comps

# Поиск цикла с помощью обхода в глубину
def find_cycle(g: Graph) -> List[int] | bool:
    mark = {v.id: 0 for v in g.vertices}
    visited = set()
    
    def dfs(p0: int):
        nonlocal S
        if len(S) > 0:
            pred = S[-1]
        else:
            pred = None
        S.append(p0)
        mark[p0] = 1
        succs = set(np.nonzero(g[p0, :])[0])
        for q in succs:
            if g.kind == GraphKind.Undirected and q == pred:
                continue
            if mark[q] == 1:
                S = S[S.index(q):]
                S.append(q)
                return True
            if mark[q] == 0:
                if dfs(q):
                    return True
        mark[p0] = 2
        S.pop()
        return False
    
    for v in [v.id for v in g.vertices]:
        S = []
        if mark[v] != 0:
            continue
        if dfs(v):
            return S

    return False


# Нахождение сильных компонент
def strong_comps(g: Graph) -> List[Set[int]]:
    R = reachability_matrix(g)
    Q = np.transpose(R)
    RQ = R * Q
    visited = set()
    comps = []
    for i in range(RQ.shape[0]):
        if i in visited: continue
        comp = set([i])
        visited.add(i)
        for j in range(RQ.shape[0]):
            if j == i: continue
            if j in visited: continue
            if np.all(RQ[j, :] == RQ[i, :]):
                comp.add(j)
                visited.add(j)
        comps.append(comp)
    return comps

### 
### Свойства
###

# Является ли граф эйлеровым
def is_euler(g: Graph) -> bool:
    if g.kind == GraphKind.Directed:
        for v in g.vertices:
            degree_in = np.count_nonzero(g[v.id, :])
            degree_out = np.count_nonzero(g[:, v.id])
            if degree_in != degree_out:
                return False
    else:
        for v in g.vertices:
            degree = np.count_nonzero(g[v.id, :])
            if degree % 2 != 0:
                return False
    return True

# Степени всех вершин графа
def degrees(g: Graph) -> list[int]:
    result = []
    for v in g.vertices:
        degree = np.count_nonzero(g[v.id, :])
        if g.kind == GraphKind.Directed:
            degree += np.count_nonzero(g[:, v.id])
        result.append(degree)
    return result

# Определение хроматического числа и раскраска графа
def coloring(g: Graph) -> Tuple[int, Dict[int, int]]:
    # Первоначальная раскраска
    colors = [v.id for v in g.vertices]
    vs = [v.id for v in g.vertices]
    coloring = {vs[0]: 0}
    for v in vs[1:]:
        conns = np.nonzero(g[v, :])[0]
        conn_colors = set()
        for v_c in conns:
            if (c := coloring.get(v_c)) is not None:
                conn_colors.add(c)
        for c in colors:
            if c not in conn_colors:
                coloring[v] = c
                break
    colors = list(set(coloring.values()))
    q = max(colors)

    def recolor_step(q: int) -> bool:
        x_i = next((v for v in vs if coloring[v] == q), None)
        if x_i is None:
            return False
        while True:
            conns = np.nonzero(g[x_i, :])[0]
            conns_before = np.where(conns < x_i)[0]
            if len(conns_before) == 0:
                return False
            x_k = np.max(conns_before)
            j_k = coloring[x_k]
            k_conns = np.nonzero(g[x_k, :])[0]
            k_conn_colors = set()
            for v_c in k_conns:
                k_conn_colors.add(coloring[v_c])
            for j_k_p in colors[j_k+1:q]:
                if j_k_p not in k_conn_colors:
                    coloring[x_k] = j_k_p
                    break
            else:
                x_i = x_k
                continue
            for v in vs[x_k+1:]:
                conns = np.nonzero(g[v, :])[0]
                conn_colors = set()
                for v_c in conns:
                    conn_colors.add(coloring[v_c])
                for c in colors[:-1]:
                    if c not in conn_colors:
                        coloring[v] = c
                        break
                else:
                    x_i = v
                    break
            else:
                return True
        return False
    
    while q > 1:
        if recolor_step(q):
            q = np.max(coloring.values())
        else:
            break
    
    return q + 1, coloring
