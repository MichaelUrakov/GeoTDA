import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import PIL
from skimage import io
from persim import plot_diagrams
from ripser import ripser, lower_star_img


fig, ax = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = ax[0][0], ax[0][1], ax[1][0], ax[1][1]

file = 'Cells.jpg'
cells_orig = plt.imread(file)
cells_gray = np.asarray(PIL.Image.fromarray(cells_orig).convert('L'))

plot_diagrams(lower_star_img(-cells_gray), ax=ax3, lifetime=True, title='Original graycale img')

smoothed = ndimage.uniform_filter(cells_gray.astype(np.float64), size=10)
smoothed += 0.1 * np.random.randn(*smoothed.shape)

dgm = lower_star_img(-smoothed)
plot_diagrams(dgm, ax=ax4, lifetime=True, title='Filtered grayscale img')

thresh = 30
idxs = np.arange(dgm.shape[0])
idxs = idxs[np.abs(dgm[:, 1] - dgm[:, 0]) > thresh]

#plt.figure(figsize=(8, 5))
ax1.imshow(cells_orig)
ax1.axis('off')
ax2.imshow(smoothed, cmap='gray')
ax2.axis('off')

X, Y = np.meshgrid(np.arange(smoothed.shape[1]), np.arange(smoothed.shape[0]))
X = X.flatten()
Y = Y.flatten()
for idx in idxs:
    bidx = np.argmin(np.abs(smoothed + dgm[idx, 0]))
    ax2.scatter(X[bidx], Y[bidx], 10, 'k')
plt.tight_layout()
plt.show()