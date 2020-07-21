import networkx as nx
from Bio import Phylo
from io import StringIO
import numpy as np

example_newick = '((A:0.1,B:0.2):0.2,(C:0.3,D:0.4):0.5);'
NUM_SITES = 5
tip_data = {'A': 'acgta', 'B': 'gcta-', 'C': 'g--a-', 'D': 'ggg-a'}
assert all(len(v) == NUM_SITES for v in tip_data.values())

## Tree parsing
G = Phylo.to_networkx(Phylo.read(StringIO(example_newick), 'newick', rooted=True))
# FIXME this just arbitrarily assigns the leaves to the first n nodes in some
# way. Need to make sure it matches up with tip_data.
print(G.nodes)
n = len(tip_data)
leaves = iter(range(n))
interior = iter(range(n, 2 * n - 1))
node_remap = {c: next(leaves if c.name is not None else interior) for c in G.nodes}
G = nx.relabel_nodes(G, mapping=node_remap)
postorder = list(nx.dfs_postorder_nodes(G))
child_parent = [None] * len(G) # the i-th entry of child_parent is the parent of node i
for k in G.pred:
    child_parent[k] = list(G.pred[k])[0] if G.pred[k] else -1

## Rate matrix
mu = .25
Q = np.full((4, 4), mu)
np.fill_diagonal(Q, -3 * mu)

## Initial partials
encoding = dict(zip('actg', np.eye(4)))
encoding['-'] = np.ones(4)
print(tip_data)
enc_tip = [np.transpose([encoding[vv] for vv in v]) for v in tip_data.values()] ## FIXME: ordering is arbitrary
print(enc_tip)

## C++ interface
# import cppyy
class MyCppyy:
    def __init__(self, outfile):
        self.f = open(outfile, 'wt')
    def add_include_path(self, path):
        pass
    def include(self, header):
        print(f"#include <{header}>", file=self.f)
    def cppdef(self, source):
        print(source, file=self.f)

cppyy = MyCppyy('eigen.cpp')

cppyy.add_include_path("/usr/include/eigen3")
cppyy.include("Eigen/Dense")
cppyy.include("algorithm")
cppyy.include("iostream")
cppyy.include("cassert")
cppyy.include("vector")
cppyy.include("unordered_map")

cppyy.cppdef('''
#include "prettyprint.hpp"

template <int num_sites>
using Partial = Eigen::Matrix<double, 4, num_sites>;

template <typename Derived>
void assign(Eigen::MatrixBase<Derived> &m, int i, int j, double x) { m(i,j) = x; }

template <int num_sites>
using PartialVector = std::vector<Partial<num_sites>, Eigen::aligned_allocator<Partial<num_sites> > >;

template <int num_sites>
using PartialNodeMap = std::unordered_map<int, Partial<num_sites>, std::hash<int>, std::equal_to<int>,
                           Eigen::aligned_allocator<std::pair<const int, Partial<num_sites> > > >;
''')

for var in 'child_parent', 'postorder':
    cppyy.cppdef('''const std::vector<int> %s = { %s };''' % (var, ",".join(map(str, locals()[var]))))

# encode partials
cppyy.cppdef(f'PartialVector<{NUM_SITES}> tip_partials = {{\n' + ",\n".join(
    f"    (Partial<{NUM_SITES}>() << %s).finished()" % ",".join(map(str, sites.reshape(-1))) for sites in enc_tip
) + '\n};')
# from cppyy.gbl import std, Eigen, assign, Partial, PartialVector
# tip_partials = PartialVector[NUM_SITES]()
# for sites in enc_tip:
#     p = Partial[NUM_SITES]()
#     for i, vec in enumerate(sites):
#         for j in range(4):
#             assign(p, j, i, vec[j])
#     tip_partials.push_back(p)

cppyy.cppdef(r'''
struct transition_matrix {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::EigenSolver<Eigen::Matrix4d> es;
    Eigen::Matrix4cd P, Pinv;
    Eigen::Vector4cd d;

    transition_matrix(const Eigen::Matrix4d Q) : es(Q), P(es.eigenvectors()),
        Pinv(P.inverse()), d(es.eigenvalues()) {}

    Eigen::Matrix4d exp_tQ (const double t)
    {
        Eigen::Matrix4d ret = (P * ((t * d).array().exp().matrix().asDiagonal()) * Pinv).real();
        return ret;
    }
};
''')

## Calculate likelihood
cppyy.cppdef('''
struct value_grad {
    Eigen::MatrixXd P_Y, grad;
};

template <int num_sites>
value_grad loglik(/*PartialVector<num_sites> tip_partials,
              std::vector<int> child_parent,
              std::vector<int> postorder,*/
              std::vector<double> times)
{
    %s
    %s
    /*
       Compute the likeliood of alignment using pruning algorithm given:

           tip_partials: vector length `n` giving partial likelihoods at each tip for each site.
           child_parent: vector mapping child nodes to parent nodes.
           postorder: postorder traversal order. It is assumed that the leaves are numbered 0, ..., `n - 1`
           times: times[i] is the branch length above node i. same length as postorder; entry for root is ignored.
    */
    const int num_nodes = postorder.size();
    const int num_leaves = tip_partials.size();
    assert(num_nodes == 2 * num_leaves - 1);
    assert(times.size() == postorder.size());

    transition_matrix trans(Q);

    PartialVector<num_sites> postorder_partials(tip_partials), preorder_partials(num_nodes);  // denoted a p, q in m.s.
    // enlarge partials to accomodate interior partials; entries 0, ..., num_leaves - 1 are the partials at the tips
    postorder_partials.resize(num_nodes);

    // create a mapping of the (two) child nodes of each parent
    int parent, child;
    std::unordered_map<int, std::pair<int, int> > parent_child;
    for (int i = 0; i < child_parent.size(); ++i)
    {
        parent = child_parent.at(i);
        if (parent == -1) continue;  // root
        std::pair<int, int> &p = parent_child[parent];
        // this will populate both entries of the pair, assuming that there are
        // only two children per parent
        p.second = p.first;
        p.first = i;
    }
    // std::cout << "parent_child: " << parent_child << std::endl;

    // create a mapping of each child's sibling, i.e. the other child of its parent
    std::unordered_map<int, int> sibling;
    for (auto &kv : parent_child)
    {
        std::pair<int, int> &p = kv.second;
        sibling[p.first] = p.second;
        sibling[p.second] = p.first;
    }


    /*** Skip for now in favor of inverting P. ***/
    // Decompose the transition matrix for later use. We are going to repeatedly solve
    // equations of the form `exp(t * Q) A = B` for B. This is equivalent
    // to `diag(exp(d * t)) P^T A = P^T B`.
    // Eigen::PartialPivLU<Partial> P_T_lu(P.transpose());

    PartialNodeMap<num_sites> Pi_pi;  // P[i] @ p[i]
    Partial<num_sites> A;

    // Compute all postorder partials
    for (int i = 0; i < postorder.size(); ++i)
    {
        parent = postorder.at(i);
        if (parent < num_leaves) continue;  // node is a leaf node; partial has already been populated
        postorder_partials.at(i) = Partial<num_sites>::Ones();
        std::pair<int, int> &p = parent_child.at(parent);
        // std::cout << "po-p parent=" << parent << " ";
        for (int child : {p.first, p.second})
        {
            // std::cout << "child=" << child << " ";
            // for each child of interior node parent, compute the transition
            // matrix along the branch of length `times[child]`.
            // We have: p_parent = P[i] * p_child = P exp(t * d) P^{-1} @ partial[child]
            Pi_pi.insert({child, trans.exp_tQ(times.at(child)) * postorder_partials.at(child)});
            // std::cout << " Pi_pi[child]=" << std::endl << Pi_pi[child] << std::endl;
            postorder_partials.at(i) = postorder_partials.at(i).cwiseProduct(Pi_pi.at(child));
        }
        // std::cout << "partial=" << std::endl << postorder_partials[i] << std::endl << std::endl;
        // std::cout << "postorder_partials: " << std::endl << postorder_partials;
    }

    // Compute all preorder partials by iterating postorder vector in reverse
    preorder_partials.at(postorder.back()).colwise() = pi;  // root node
    for (int i = postorder.size() - 2; i >= 0; --i)
    {
        child = postorder.at(i);
        parent = child_parent.at(child);
        // Equation (7)
        A = preorder_partials.at(parent).cwiseProduct(Pi_pi.at(sibling.at(child)));
        // preorder_partials[child] = q[i] = P[i]^T @ A = Pinv.T @ e^(t[i] d) @ P.T @ A
        preorder_partials.at(child) = trans.exp_tQ(times.at(child)).transpose() * A;
        /* std::cout << "preorder_partials child=" << child << " sibling=" << sibling[child] << 
                      " Pi_pi[sibling[child]]=" << Pi_pi[sibling[child]] << " q[child]=" << 
                      std::endl << preorder_partials[child] << std::endl; */
    }

    // finally compute gradients from equation 9
    value_grad ret;
    ret.P_Y = pi.transpose() * postorder_partials.back();  // the overall likelihood is pi * the root partial
    ret.grad.resize(num_nodes, num_sites);
    for (int i = 0; i < num_nodes; ++i)
        ret.grad.row(i) = ((times[i] * Q.transpose()) * preorder_partials.at(i)).cwiseProduct(postorder_partials.at(i)).colwise().sum();
    ret.grad *= (1. / ret.P_Y.array()).matrix().asDiagonal();
    return ret;
}
''' % (
    'Eigen::Vector4d pi; pi << .25, .25, .25, .25;',
    'Eigen::Matrix4d Q; Q << %s;' % ",".join(map(str, Q.reshape(-1)))
))

cppyy.cppdef('''
int main(int argc, char** argv)
{{
    std::vector<double> times = {{ {times} }};
    value_grad ll = loglik<{NUM_SITES}>(/*tip_partials, child_parent, postorder,*/ times);
    std::cout << ll.P_Y << std::endl;
    std::cout << ll.grad << std::endl;

}}
'''.format(NUM_SITES=NUM_SITES, times=",".join(["1."] * (2 * n - 1)))
)

## make up some times to evaluate
# n = len(tip_data)
# times = [1.] * (2 * n - 2)
# assert len(times) == len(postorder)
# cppyy.gbl.loglik(tip_partials, child_parent, postorder, times)
