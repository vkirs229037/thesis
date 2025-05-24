from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum, auto
import numpy as np
import igraph as ig

class GraphKind(Enum):
    Directed = auto()
    Undirected = auto()

@dataclass
class Vertex:
    name: str
    label: str
    id: int

    def __str__(self):
        return self.name
    
    def __hash__(self):
        return self.id

class Graph:
    def __init__(self, vertices: List[Vertex], vertex_connections: Dict[Tuple[int, int], int], kind: GraphKind):
        self.n = len(vertices)
        self.adj_matrix = np.zeros((self.n, self.n), dtype=np.dtype(int))
        self.vertices = sorted(vertices, key=lambda v: v.id)
        self.edges = vertex_connections
        for pair, weight in self.edges.items():
            self.adj_matrix[pair] = weight
            if kind == GraphKind.Undirected:
                self.adj_matrix[pair[1], pair[0]] = weight
        if kind == GraphKind.Undirected:
            es = {}
            for kvp in self.edges.items():
                es[(kvp[0][1], kvp[0][0])] = kvp[1]
            self.edges.update(es)
        self.kind = kind

    def to_ig_graph(self):
        match self.kind:
            case GraphKind.Directed:
                g = ig.Graph(self.n, self.edges, directed=True)
                g.vs["name"] = list(map(lambda v: v.name, self.vertices))
                g.vs["label"] = list(map(lambda v: v.label, self.vertices))
                g.es["weight"] = list(map(lambda w: str(w), self.edges.values()))
                return g
            case GraphKind.Undirected:
                es = []
                for edge in self.edges:
                    if (edge[1], edge[0]) in self.edges and self.adj_matrix[edge[0], edge[1]] == self.adj_matrix[edge[1], edge[0]]:
                        if edge[0] < edge[1]:
                            es.append(edge)
                    else:
                        es.append(edge)
                g = ig.Graph(self.n, es, directed=False)
                g.vs["name"] = list(map(lambda v: v.name, self.vertices))
                g.vs["label"] = list(map(lambda v: v.label, self.vertices))
                g.es["weight"] = list(map(lambda w: str(w), self.edges.values()))
                return g

    def print(self):
        for v in self.vertices:
            print(v)
        width = len(max(self.vertices, key=lambda v: len(v.name)).name) if len(self.vertices) > 0 else 1
        print(" ", end="")
        v_format = f"{{:>{width}}}"
        for v in self.vertices:
            print(v_format.format(v.name), end=" ")
        print("\n", end="")
        n = self.adj_matrix.shape[0]
        for i in range(0, n):
            print(v_format.format(self.vertices[i].name), end=" ")
            for j in range(0, n):
                print(v_format.format(self.adj_matrix[i, j]), end=" ")
            print("\n", end="")

    def __getitem__(self, tup):
        return self.adj_matrix[tup]
    
    def __setitem__(self, tup, val):
        self.adj_matrix[tup] = val