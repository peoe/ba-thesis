# taken from https://pythonot.github.io/auto_examples/plot_OT_1D_smooth.html

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

bins = 100

# bin positions
x = np.arange(bins, dtype=np.float64)

# Gaussian distributions
a = gauss(bins, m=20, s=5)
b = gauss(bins, m=60, s=10)

# loss matrix
M = ot.dist(x.reshape((bins, 1)), x.reshape((n, 1)))
M /= M.max()