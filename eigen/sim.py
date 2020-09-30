from util import make_header, run_stan
    
data = make_header('sim.tree', 'sim.fa', 'fasta', 'dates.csv', 'sim.hpp')
data['m'] = 10
data['x1'] = 0
fit = run_stan('sim.hpp', 'vbsky_fix_rate.stan', data, 1, 'sim')