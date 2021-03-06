#!/usr/bin/env python
import utils
import os
import subprocess
import argparse

from subprocess import check_call

parser = argparse.ArgumentParser()
parser.add_argument('alignment')
parser.add_argument('trees')
args = parser.parse_args()

dir, file = os.path.split(args.alignment)

trees_path = os.path.join(dir, 'output.trees')
temp_tree_path = os.path.join(dir, 'temp.tree')

trees_path = args.trees
# utils.convert_trees(args.trees, trees_path)

my_path = os.path.split(os.path.realpath(__file__))[0]

i = 0

with open(trees_path, 'r') as f:
    for line in f:
        line = line.rstrip('\n').rstrip('\r')

        if i != 872:
            i += 1
            continue

        with open(temp_tree_path, 'w') as f2:
            f2.write(line)

        output = args.alignment + str(i) + '.log'

        if os.path.lexists(output+'.diag'):
            i += 1
            continue

        # cmd = ['python', os.path.join(my_path, 'phylostan.py'), '-t', temp_tree_path, '-i', args.alignment, '-o', output, '-m', 'JC69']
        cmd = ['python', os.path.join(my_path, 'phylostan.py'), '-t', temp_tree_path, '-i', args.alignment, '-o', output,
               '-m', 'GTR', '-q', 'fullrank']
        print(' '.join(cmd))

        # Stan can fail so we just restart it until success
        done = False
        while not done:
            done = True
            try:
                check_call(cmd)
            except subprocess.CalledProcessError, e:
                done = False
        i += 1
