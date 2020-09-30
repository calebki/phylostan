import networkx as nx
from Bio import Phylo, SeqIO
import numpy as np
import pystan
import jinja2
import scipy.sparse
import csv

def get_dates(dates_file):
    reader = csv.reader(open(dates_file, 'r'))
    node_dates_dict = {rows[0]:float(rows[1]) for rows in reader if rows[0].isnumeric()}
    return node_dates_dict

def get_lowers(dates, postorder, G):
    n = int((len(postorder) + 1)/2)
    lowers = [0]

    max_date = max(dates)
    min_date = min(dates)

    times = [None] * n

    #time is going backwards
    if min_date == 0:
        times = dates
        oldest = max_date

    #time is going forwards
    else:
        for i in range(n):
            times[i] = max_date - dates[i]
        oldest = max_date - min_date

    lowers = [0] * (2*n - 1)

    for i in postorder:
        if i < n:
            lowers[i] = times[i]
        else:
            lowers[i] = max([lowers[j] for j in G.succ[i]])

    return lowers, oldest

def make_header(tree_file, aln_file, aln_file_type, dates_file, out_file):
    tip_data = SeqIO.to_dict(SeqIO.parse(aln_file, aln_file_type))
    NUM_SITES = len(next(iter(tip_data.values())))
    assert all(len(v) == NUM_SITES for v in tip_data.values())

    dates = get_dates(dates_file)

    ## Tree parsing
    G = Phylo.to_networkx(Phylo.read(tree_file, 'newick', rooted=True))
    # FIXME this just arbitrarily assigns the leaves to the first n nodes in some
    # way. Need to make sure it matches up with tip_data.
    print(G.nodes())
    n = len(tip_data)
    leaves = iter(range(n))
    interior = iter(range(n, 2 * n - 1))
    node_remap = {}
    tip_remap = {}
    dates_remap = [None] * n

    for c in nx.dfs_postorder_nodes(G):
        if c.name is not None:
            node_id = next(leaves)
            node_remap[c] = node_id
            tip_remap[node_id] = tip_data[c.name]
            dates_remap[node_id] = dates[c.name]
        else:
            node_remap[c] = next(interior)
    G = nx.relabel_nodes(G, mapping=node_remap)
    postorder = list(nx.dfs_postorder_nodes(G))
    preorder_map = np.empty((len(G),2), dtype=int)
    preorder_map[:,0] = list(nx.dfs_preorder_nodes(G))
    for i in range(len(G)):
        preorder_map[i,1] = list(G.pred[preorder_map[i,0]])[0] if list(G.pred[preorder_map[i,0]]) else 0
    preorder_map += 1
    child_parent = [None] * len(G) # the i-th entry of child_parent is the parent of node i
    for k in G.pred:
        child_parent[k] = list(G.pred[k])[0] if G.pred[k] else -1

    lowers, oldest = get_lowers(dates_remap, postorder, G)

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
    sparse_tip_partials = []
    for i in range(n):
        v = tip_remap[i]
        tip_partials.append(np.transpose([encoding[vv.lower()] for vv in v])) ## FIXME: ordering is arbitrary
        sp = scipy.sparse.coo_matrix(tip_partials[-1])
        sparse_tip_partials.append(zip(sp.row, sp.col, sp.data))
    print(tip_partials)

    ## Write C++ header file
    hpp = jinja2.Template(open('eigen.j2', 'rt').read())
    with open(out_file, "wt") as out:
        s = hpp.render(child_parent=child_parent, postorder=postorder, Q=Q, pi=pi, 
                       tip_partials=tip_partials, sparse_tip_partials=sparse_tip_partials, 
                       num_sites=NUM_SITES)
        out.write(s)
    data = {'S':n, 'L':NUM_SITES, 'map':preorder_map, 'rate':1.0, 'lower_root':max(oldest, 0.0), 'lowers': lowers, 'sample_times':dates_remap}
    return data

def run_stan(header, stan_file, data, seed, out):
    sample_path = f'output/{out}_{seed}'
    # tree_path = f'{sample_path}.trees'

    include_files = [header, "prune_stan.hpp"]

    sm = pystan.StanModel(file=stan_file,
                          allow_undefined=True,
                          includes=include_files,
                          include_dirs=["."],

                        #   verbose=True,
                        #   extra_compile_args=["--std=c++14", "-Wno-int-in-bool-context"]  # "-Wfatal-errors", 
                          )

    fit = sm.vb(data=data, iter=1000, algorithm='meanfield', seed = seed, sample_file=sample_path)
    return fit