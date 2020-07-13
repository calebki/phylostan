import pystan
import numpy as np
import os
import time

theta = 10
n = 50
y = np.random.exponential(10, n)
data = {'n':y.size ,'y': y}

seed = 1

include_files = ["external_manual.hpp"]
start = time.time()
sm = pystan.StanModel(file="model.stan",
                      allow_undefined=True,
                      includes=include_files,
                      include_dirs=["."],
                      verbose=True
                      )
print("Compilation took {0:.1f} seconds".format(time.time() - start))
fit = sm.vb(data=data, iter=1000, algorithm='meanfield', seed = seed)
print(fit['mean_pars'])