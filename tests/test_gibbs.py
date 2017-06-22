import cg
import numpy as np

code = ['4ake','1oelA','1tyq'][2]

x = cg.load_example(code)
K = {'4ake': 50, '1oelA': 150, '1tyq': 500}[code]
k = 20

gibbs   = cg.GibbsSampler(x, K, k, run_kmeans=True)
samples = gibbs.run(1e4)

## show results

cg.utils.plot_samples(samples)
