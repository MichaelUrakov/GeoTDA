import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import PIL
from skimage import io, color
import ripser as rips
import persim as pers
import skimage

rs = 10
thresh = 30
size = 20
noise_coeff = 0.5
file = "piece_of_core_jpg.jpg"

np.random.seed(rs)

fig, ax = plt.subplots(2, 1)

img_orig = plt.imread(file)
img = np.asarray(PIL.Image.fromarray(img_orig).convert('L'))

img = ndimage.uniform_filter(img.astype(np.float64), size=size)
img += noise_coeff * np.random.randn(*img.shape)

ax[0].imshow(img_orig, cmap='gray')
ax[1].imshow(img, cmap='gray')

dgm = rips.lower_star_img(-img)

idxs = np.arange(dgm.shape[0])
idxs = idxs[np.abs(dgm[:, 1] - dgm[:, 0]) > thresh]

X, Y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
X, Y = X.flatten(), Y.flatten()

pointcloud = []
for i in idxs:
    bidx = np.argmin(np.abs(img + dgm[i, 0]))
    ax[1].scatter(X[bidx], Y[bidx], 5, 'k')
    #pointcloud.append([X[bidx], Y[bidx]])
ax[0].axis('off')
ax[1].axis('off')
#ax[2].axis('off')
#pointcloud = np.array(pointcloud)
#dgm_res = rips.ripser(pointcloud, maxdim=2)['dgms']
#pers.plot_diagrams(dgm_res, ax=ax[2], legend=False)
plt.title(f'rs = {rs}, t = {thresh}, s = {size}, nc = {noise_coeff}', fontsize=14)
plt.tight_layout()

