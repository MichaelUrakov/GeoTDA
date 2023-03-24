import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
import tadasets
import ripser
import persim



np.random.seed(565656)

data_clean = tadasets.dsphere(d=1, n=100, noise=0.0)
data_noisy = tadasets.dsphere(d=1, n=100, noise=0.10) 

#data_clean = tadasets.infty_sign(n=100, noise=0.0)
#data_noisy = tadasets.infty_sign(n=100, noise=0.02) 

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0][0].scatter(data_clean[:, 0], data_clean[:, 1], label='clean', s=10)
ax[0][1].scatter(data_noisy[:, 0], data_noisy[:, 1], label='noisy', s=15)
ax[0][0].legend()
ax[0][1].legend()
ax[0][0].axis('equal')
ax[0][1].axis('equal')

#rips = ripser.Rips(maxdim=2)

#dgm_clean = rips.transform(data_clean)
dgm_clean = ripser.ripser(data_clean, maxdim=1)['dgms']
persim.plot_diagrams(dgm_clean, ax=ax[1][0])
#rips.plot(ax=ax[1][0])

#dgm_noisy = rips.transform(data_noisy)
rips_noisy = ripser.ripser(data_noisy, maxdim=1)
dgm_noisy = rips_noisy['dgms']
persim.plot_diagrams(dgm_noisy, ax=ax[1][1])
#rips.plot(ax=ax[1][1])
