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
        if self.label == "":
            return f"[ID {self.id}] {self.name}" 
        return f"[ID {self.id}] {self.name}: {self.label}"
    
    def __hash__(self):
        return self.id

class Graph:
    def __init__(self, vertices: List[Vertex], vertex_connections: Dict[Tuple[int, int], int], kind: GraphKind):
        self.n = len(vertices)
        self.incidence_matrix = np.zeros((self.n, self.n), dtype=np.dtype(int))
        self.vertices = sorted(vertices, key=lambda v: v.id)
        self.edges = vertex_connections
        for pair, weight in self.edges.items():
            self.incidence_matrix[pair[0], pair[1]] = weight
        self.kind = kind

    def to_ig_graph(self):
        n = len(self.vertices)
        edges = [(v1.id, v2.id) for v1 in self.vertices for v2 in self.vertices if self.incidence_matrix[v1.id, v2.id] != 0]
        print(edges)
        edges_filtered = []
        for edge in edges:
            if (edge[1], edge[0]) in edges and self.incidence_matrix[edge[0], edge[1]] == self.incidence_matrix[edge[1], edge[0]]:
                if edge[0] < edge[1]:
                    edges_filtered.append(edge)
            else:
                edges_filtered.append(edge)
        print(edges_filtered)
        g = ig.Graph(n, edges_filtered, directed=True)
        g.vs["name"] = list(map(lambda v: v.name, self.vertices))
        g.vs["label"] = list(map(lambda v: v.label, self.vertices))
        g.es["weight"] = list(map(lambda pair: str(self.incidence_matrix[pair[0], pair[1]]), edges_filtered))
        print(g.es["weight"])
        g.es["directed"] = list(map(lambda pair: self.incidence_matrix[pair[0], pair[1]] != self.incidence_matrix[pair[1], pair[0]], edges_filtered))
        return g
    
    def to_ig_graph(self):
        match self.kind:
            case GraphKind.Directed:
                g = ig.Graph(self.n, self.edges, directed=True)
                g.vs["name"] = list(map(lambda v: v.name, self.vertices))
                g.vs["label"] = list(map(lambda v: v.label, self.vertices))
                g.es["weight"] = list(map(lambda w: str(w), self.edges.values()))
                return g
            case GraphKind.Undirected:
                g = ig.Graph(self.n, self.edges, directed=False)
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
        n = self.incidence_matrix.shape[0]
        for i in range(0, n):
            print(v_format.format(self.vertices[i].name), end=" ")
            for j in range(0, n):
                print(v_format.format(self.incidence_matrix[i, j]), end=" ")
            print("\n", end="")

    def __getitem__(self, tup):
        return self.incidence_matrix[tup]