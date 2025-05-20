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

# Алгоритм Флойда
def floyd(g: Graph) -> Tuple[np.ndarray, np.ndarray] | bool:
    c = copy.deepcopy(g[:, :])
    vs = np.array(list(map(lambda v: v.id, g.vertices)))
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
    all_edges = list(g.edges.keys())
    marked = {}
    result = []
    first_edge = random.choice(all_edges)
    marked[first_edge] = True
    result.append(first_edge)
    while set(marked.keys()) != set(all_edges):
        e = result[-1]
        v = e[1]
        conns = np.nonzero(g[v, :])[0]
        edges = set([(v, w) for w in conns if marked.get((v, w)) != True])
        bridges = set()
        for w in conns:
            if v == w:
                continue
            c = g.edges[(v, w)]
            del g.edges[(v, w)]
            g[v, w] = 0
            r_m = reachability_matrix(g)
            if r_m[v, w] == 0:
                # Нашли мост
                bridges.add((v, w))
            g.edges[(v, w)] = c
            g[v, w] = c
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
    marked = {}
    result = []
    first_edge = random.choice(all_edges)
    marked[first_edge] = True
    marked[(first_edge[1], first_edge[0])] = True
    result.append(first_edge)
    while set(marked.keys()) != set(all_edges):
        e = result[-1]
        v = e[1]
        conns = np.nonzero(g[v, :])[0]
        edges = set([(v, w) for w in conns if marked.get((v, w)) != True or marked.get((w, v)) != True])
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
        marked[new_e] = True
        marked[new_e[1], new_e[0]] = True
    return result


# Задача китайского почтальона
def chinesepostman(g: Graph) -> List[int]:
    raise NotImplementedError

# Общий обход графа с запоминанием вершин
def walk(g: Graph, p0: int) -> List[int]:
    r_m = reachability_matrix(g)
    visited = set([p0])
    path = [p0]
    S = [p0]
    while len(S) > 0:
        q = S.pop()
        succs = set(np.nonzero(r_m[q, :])[0])
        for p in succs:
            if p not in visited:
                visited.add(p)
                S.append(p)
                path.append(p)
    return path

# Нахождение компонент связности
def conn_comps(g: Graph) -> List[Set[int]]:
    visited = set()
    comps = []
    for v in [v.id for v in g.vertices]:
        if v in visited:
            continue
        path = walk(g, v)
        visited.update(path)
        comps.append(set(path))    
    return comps

### 
### Свойства
###

# Является ли граф эйлеровым
def is_euler(g: Graph) -> bool:
    if g.kind == GraphKind.Directed:
        for v in g.vertices:
            degree_in = np.sum(np.where(g[v.id, :] != 0, 1, 0))
            degree_out = np.sum(np.where(g[:, v.id] != 0, 1, 0))
            if degree_in != degree_out:
                return False
    else:
        for v in g.vertices:
            degree = np.sum(np.where(g[v.id, :] != 0, 1, 0))
            if degree % 2 != 0:
                return False
    return True

# Степени всех вершин графа
def degrees(g: Graph) -> list[int]:
    result = []
    for v in g.vertices:
        degree = np.sum(np.where(g[v.id, :] != 0, 1, 0))
        if g.kind == GraphKind.Directed:
            degree += np.sum(np.where(g[:, v.id] != 0, 1, 0))
        result.append(degree)
    return result

# Определение хроматического числа и раскраска графа
def chrom_num(g: Graph) -> Tuple[int, Dict[int, int]]:
    # Первоначальная раскраска
    ret_flag = False
    colors = [v.id for v in g.vertices]
    vs = [v.id for v in g.vertices]
    coloring = {vs[0]: 0}
    for v in vs[1:]:
        conns = np.nonzero(g[v, :])[0]
        conn_colors = {}
        for v_c in vs:
            if (c := coloring.get(v)) is not None and g[v, v_c] != 0:
                conn_colors.add(c)
        for c in colors:
            if c not in conn_colors:
                coloring[v] = c
                break
    q = max(coloring.values())
    cr_num = q + 1
    return cr_num, coloring
    print(f"initial q = {q}")
    ret_v = np.min(color_to_v[q])
    j_k = q
    while ret_v != vs[0]:
        print("ret_v", ret_v)
        print("all_cons", np.nonzero(g[ret_v, :])[0])
        conns = np.array(list(filter(lambda v: v < ret_v, np.nonzero(g[ret_v, :])[0])))
        print("conns", conns)
        if len(conns) == 0:
            break
        rec_v = np.max(conns)
        print("rec_v", rec_v)
        rec_conns = set(np.nonzero(g[ret_v, :])[0])
        rec_c = v_to_color[rec_v]
        for c in colors[rec_c+1:]:
            if len(set(color_to_v[c]).intersection(rec_conns)) == 0:
                if c == q:
                    print(f"получен j_k {c} == q {q}")
                    j_k = q
                    break
                print(f"найден j_k = {c}, перекраска x_k ({rec_v}) в него")
                j_k = c
                color_to_v[c].append(rec_v)
                color_to_v[c].sort()
                color_to_v[v_to_color[rec_v]].remove(rec_v)
                v_to_color[rec_v] = c
                break
        else:
            j_k = q
        if j_k < q:
            print("j_k & q", j_k, q)
            for v in vs[rec_v+1:]:
                if ret_flag:
                    break
                conns = set(np.nonzero(g[v, :])[0])
                for c in colors:
                    if len(set(color_to_v[c]).intersection(conns)) == 0:
                        if c == q:
                            j_k = q
                            print(f"x_i {v} требует q, возврат из нее")
                            ret_v = v
                            ret_flag = True
                            print("color_to_v", color_to_v)
                            print("v_to_color", v_to_color)
                            break
                        print(f"перекраска v {v} в цвет {c}")
                        color_to_v[c].append(v)
                        color_to_v[c].sort()
                        color_to_v[v_to_color[v]].remove(v)
                        v_to_color[v] = c
                        break
            for _q in color_to_v.keys():
                if color_to_v[_q] == []:
                    q = _q - 1
                    colors = colors[:q + 1]
                    break
            print(f"новый q = {q}")
        else:
            print(f"j_k ({j_k}) == q ({q}) или не найден подходящий j_k, ret_v ({ret_v}) = rec_v (x_k = {rec_v})")
            print("color_to_v", color_to_v)
            print("v_to_color", v_to_color)
            ret_v = rec_v
            continue
        if not ret_flag:
            print("reached end of loop, ret_v = color_to_v[q][-1]")
            print("color_to_v", color_to_v)
            print("v_to_color", v_to_color)
            ret_v = np.min(color_to_v[q])
        else:
            ret_flag = False
    print(color_to_v)
    return q+1, v_to_color