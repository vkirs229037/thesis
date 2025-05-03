import sys, random

arg = ""
try:
    arg = sys.argv[1] 
except:
    arg = "random"

if arg == "random":
    n = random.randint(6, 33)
    verts = range(n)
    edges = []
    for v1 in verts:
        for v2 in verts:
            edges.append((v1, v2))
    random.shuffle(edges)
    for pair in edges:
        if random.random() < 0.4:
            continue
        op = ""
        if random.random() >= 0.5:
            op = "-"
        else:
            op = ">"
        print(pair[0], op, pair[1], ";")