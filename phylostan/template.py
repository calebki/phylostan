import dendropy
import numpy
import pystan
import pickle
from dendropy import Tree, DnaCharacterMatrix
import os
import sys
from generate_script import get_model
import utils

epi = 'sanity'
stem = f'../examples/{epi}/' #Stem for files

_model = 'HKY'
_tree = f'{stem}{epi}.tree'
_input = f'{stem}{epi}.fa' #sequence file
_heterochronous = True
_clock = 'strict' #choices=['strict', 'ace', 'acln', 'acg', 'aoup', 'ucln', 'uced', 'gmrf', 'hsmrf'], Type of clock
_estimate_rate = False #Takes T/F values 
_rate = 1.0 #substitution rate
_lower_root = 0.0 # default=0.0 Lower bound of the root
if epi == 'sim':
    _dates = f'{stem}dates.csv'
elif epi =='sanity':
    _dates = None
else:
    _dates = 'fasta' #Comma-separated (csv) file containing sequence dates with header 'name,date'
_categories = 1 #Number of categories default 1
_invariant = 'weibull' #choices=['weibull', 'discrete'], default='weibull' Weibull or discrete distribution to model rate heterogeneity across sites
tree_prior = 'bdsky'
_script = f'skyride-HKY.stan' #STAN script files
_compile = True # 'action="store_true", help="""Compile Stan script"""
_algorithm = 'vb' #algorithm choices=['vb', 'nuts', 'hmc'] default='vb'
_iter = 100000 #Maximum number of iterations for variational inference or number of iterations for NUTS and HMC algorithms

#vb parameters
_samples = 10000 #Number of samples to be drawn from the variational distribution (variational only)
_eta = None #Not required eta for Stan script (variational only)
_seed = 6 #Seed for Stan script
_tol_rel_obj = 0.001 #Convergence tolerance on the relative norm of the objective, defaults to 0.001 (variational only)
_elbo_samples = 100 #Number of samples for Monte Carlo estimate of ELBO (variational only) default 100
_grad_samples = 1 #Number of samples for Monte Carlo estimate of gradients (variational only) default 1
_variational = 'meanfield' #choices=['meanfield', 'fullowerslrank'], default='meanfield', Variational distribution family

''' skygrid parameters
_cutoff = 
_grid = 
'''

#HMC Nuts only
_chains = 1
_thin = 1

taxa = dendropy.TaxonNamespace()
tree_format = 'newick'
with open(_tree) as fp:
    if next(fp).upper().startswith('#NEXUS'):
        tree_format = 'nexus'

tree = Tree.get_from_path(src=_tree, schema=tree_format, tree_offset=0, taxon_namespace=taxa, preserve_underscores=True,
                rooting='force-rooted')

tree.resolve_polytomies(update_bipartitions=True)

utils.setup_indexes(tree)

oldest = utils.setup_dates(tree, _dates, _heterochronous)

peeling = utils.get_peeling_order(tree)
sequence_count = len(tree.taxon_namespace)
data = {'peel': peeling, 'S': sequence_count}

if _input:
    seqs_args = dict(schema='nexus', preserve_underscores=True)
    with open(_input) as fp:
        if next(fp).startswith('>'):
            seqs_args = dict(schema='fasta')

    dna = DnaCharacterMatrix.get(path=_input, taxon_namespace=taxa, **seqs_args)
    alignment_length = dna.sequence_size
    sequence_count = len(dna)
    if sequence_count != len(dna.taxon_namespace):
        sys.stderr.write('taxon names in trees and alignment are different')
        exit(2)

    print('Number of sequences: {} length {} '.format(sequence_count, alignment_length))
    print('Model: ' + _model)

    tipdata, weights = utils.get_dna_leaves_partials_compressed(dna)
    alignment_length = len(weights)

    data.update({'tipdata': tipdata, 'L': alignment_length, 'weights': weights})

if _clock is not None:
		data['map'] = utils.get_preorder(tree)
		if not _estimate_rate:
			data['rate'] = _rate if _rate else 1.0
		if _heterochronous:
			data['lowers'] = utils.get_lowers(tree)
			data['lower_root'] = max(oldest, _lower_root)
		else:
			data['lower_root'] = _lower_root
else:
    last = peeling[-1]
    if last[0] > last[1]:
        peeling[-1] = [last[1], last[0], last[2]]

if _categories > 1:
    data['C'] = _categories
    if _invariant:
        data['C'] += 1

if _clock is not None:
    if tree_prior == 'skygrid':
        data['G'] = _grid - 1
        data['grid'] = numpy.linspace(0, _cutoff, _grid)[1:]
    elif tree_prior == 'skyride':
        # number of coalescent intervals
        data['I'] = sequence_count - 1
    elif tree_prior == 'bdsky':
        data['m'] = 2
        data['sample_times'] = data['lowers'][:sequence_count]

if _model == 'GTR':
    data['frequencies_alpha'] = [1, 1, 1, 1]
    data['rates_alpha'] = [1, 1, 1, 1, 1, 1]
elif _model == 'HKY':
    data['frequencies_alpha'] = [1, 1, 1, 1]

# Samples output file
sample_path = f'{stem}{epi}'
tree_path = f'{sample_path}.trees'

binary = _script.replace('.stan', '.pkl')
if binary == _script:
    binary = _script + '.pkl'
if not os.path.lexists(binary) or _compile:
    sm = pystan.StanModel(file=_script)
    with open(binary, 'wb') as f:
        pickle.dump(sm, f)
else:
    sm = pickle.load(open(binary, 'rb'))

if _algorithm == 'vb':
    stan_args = {}
    stan_args['output_samples'] = _samples
    if _eta:
        stan_args['eta'] = _eta
        stan_args['adapt_engaged'] = False
    if _seed:
        stan_args['seed'] = _seed

    fit = sm.vb(data=data, tol_rel_obj=_tol_rel_obj, elbo_samples=_elbo_samples, grad_samples=_grad_samples,
                iter=_iter, sample_file=sample_path, diagnostic_file=sample_path + ".diag",
                algorithm=_variational, **stan_args)

    # parse the log file
    utils.convert_samples_to_nexus(tree, sample_path, tree_path, _rate)
    utils.parse_log(sample_path, 0.05)
else:
    stan_args = {'seed': _seed}
    fit = sm.sampling(data=data, algorithm=_algorithm.upper(), sample_file=sample_path, chains=_chains,
                        iter=_iter, thin=_thin, **stan_args)

    # chain=1 pystan uses sample_file
    if _chains == 1:
        if sample_path.endswith('.csv'):
            tree_path = sample_path.replace('.csv', '.trees')
        utils.convert_samples_to_nexus(tree, sample_path, tree_path, _rate)
        utils.parse_log(sample_path, 0.05)
    # chain>1 pystan appends _{chain}.csv to sample_file
    else:
        for chain in range(_chains):
            sample_path_chain = sample_path + '_{}.csv'.format(chain)
            tree_path_chain = sample_path + '_{}.trees'.format(chain)
            utils.convert_samples_to_nexus(tree, sample_path_chain, tree_path_chain, _rate)
            utils.parse_log(sample_path_chain, 0.05)