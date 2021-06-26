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
a = gauss(bins, m=40, s=10)
bs = []
bs.append(gauss(bins, m=80, s=4))

# uniform distribution
bs.append(0.01 * np.ones(bins))

# loss matrix
M = ot.dist(x.reshape((bins, 1)), x.reshape((bins, 1)))
M /= M.max()

k = 1
im = 1
for b in bs:
#	print(a, ' ', type(a), ' ', np.size(a))
#	print(b, ' ', type(b), ' ', np.size(b))

	# Source and Target
	pl.figure(k, figsize=(6.4, 3))
	pl.plot(x, a, 'b', label='Source distribution')
	pl.plot(x, b, 'r', label='Target distribution')
	pl.legend()
	fname = 'images/regVSunreg/dist' + str(im) + 'ToGaussianFig' + str(k) + '.pdf'
	pl.savefig(fname)
	k = k + 1

	# EMD
	G0 = ot.emd(a, b, M)
	pl.figure(k, figsize=(5, 5))
	ot.plot.plot1D_mat(a, b, G0, 'OT matrix G0')
	fname = 'images/regVSunreg/dist' + str(im) + 'ToGaussianFig' + str(k) + '.pdf'
	pl.savefig(fname)
	k = k + 1

	# Reg
	lambdas = [1, 1e-1, 1e-2, 1e-3, 2e-4]
	for lambd in lambdas:
		Gsm = ot.smooth.smooth_ot_dual(a, b, M, lambd, reg_type='kl')
		pl.figure(k, figsize=(5, 5))
		title = 'OT matrix Smooth OT ent. reg. ' + str(lambd)
		ot.plot.plot1D_mat(a, b, Gsm, title=title)
		fname = 'images/regVSunreg/dist' + str(im) + 'ToGaussianFig' + str(k) + '.pdf'
		pl.savefig(fname)
		k = k + 1

	im = im + 1