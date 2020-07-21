import networkx as nx
from Bio import Phylo, SeqIO
from io import StringIO
import numpy as np
import pystan
import jinja2

tree_file = 'example.tree'
tip_data = SeqIO.to_dict(SeqIO.parse("example.nexus", "nexus"))
NUM_SITES = len(next(iter(tip_data.values())))
assert all(len(v) == NUM_SITES for v in tip_data.values())

## Tree parsing
G = Phylo.to_networkx(Phylo.read(tree_file, 'newick', rooted=True))
# FIXME this just arbitrarily assigns the leaves to the first n nodes in some
# way. Need to make sure it matches up with tip_data.
print(G.nodes)
n = len(tip_data)
leaves = iter(range(n))
interior = iter(range(n, 2 * n - 1))
node_remap = {}
tip_remap = {}
for c in G.nodes():
    if c.name is not None:
        node_id = next(leaves)
        node_remap[c] = node_id
        tip_remap[node_id] = tip_data[c.name]
    else:
        node_remap[c] = next(interior)
G = nx.relabel_nodes(G, mapping=node_remap)
postorder = list(nx.dfs_postorder_nodes(G))
child_parent = [None] * len(G) # the i-th entry of child_parent is the parent of node i
for k in G.pred:
    child_parent[k] = list(G.pred[k])[0] if G.pred[k] else -1

## Rate matrix
mu = .25
Q = np.full((4, 4), mu)
np.fill_diagonal(Q, -3 * mu)
pi = np.ones(4) / 4

## Initial partials
encoding = dict(zip('actg', np.eye(4)))
encoding['-'] = np.ones(4)
print(tip_data)
tip_partials = []
for i in range(n):
    v = tip_remap[i]
    tip_partials.append(np.transpose([encoding[vv.lower()] for vv in v])) ## FIXME: ordering is arbitrary
print(tip_partials)

## Write C++ header file
hpp = jinja2.Template(open('eigen.j2', 'rt').read())
with open("eigen.hpp", "wt") as out:
    s = hpp.render(child_parent=child_parent, postorder=postorder, Q=Q, pi=pi, tip_partials=tip_partials, num_sites=NUM_SITES)
    out.write(s)

# encode partials
data = {'S':n, 'L':NUM_SITES, 'map':child_parent, 'rate':1.0, 'lower_root':0.0}
seed = 1

include_files = ["eigen.hpp"]

sm = pystan.StanModel(file="example.stan",
                      allow_undefined=True,
                      includes=include_files,
                      include_dirs=["."],
                      verbose=True,
                      extra_compile_args=["--std=c++14", "-Wno-int-in-bool-context"]  # "-Wfatal-errors", 
                      )
fit = sm.vb(data=data, iter=1000, algorithm='meanfield', seed = seed)
print(fit['mean_pars'])
