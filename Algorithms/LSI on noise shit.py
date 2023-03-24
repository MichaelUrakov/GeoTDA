import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import PIL
from skimage import io, color
import ripser as rips
import persim as pers


#np.random.seed(2330423)

fig, ax = plt.subplots(1, 3)


img_orig = np.random.randn(20, 20) * 100


#img = img > 0.1
#img = img.astype(float)
img = ndimage.uniform_filter(img_orig.astype(np.float64), size=2)
#img += 0.1 * np.random.randn(*img.shape)
ax[0].imshow(img_orig)

dgm = rips.lower_star_img(-img)


#pers.plot_diagrams(dgm, ax=ax[1])
ax[1].imshow(img)

thresh = 10

idxs = np.arange(dgm.shape[0])
idxs = idxs[np.abs(dgm[:, 1] - dgm[:, 0]) > thresh]

X, Y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
X, Y = X.flatten(), Y.flatten()

pointcloud = []


for i in idxs:
    bidx = np.argmin(np.abs(img + dgm[i, 0]))
    ax[1].scatter(X[bidx], Y[bidx], 10, 'k')
    pointcloud.append([X[bidx], Y[bidx]])


ax[0].axis('off')
ax[1].axis('off')
pointcloud = np.array(pointcloud)
dgm_res = rips.ripser(pointcloud, maxdim=2)['dgms']
pers.plot_diagrams(dgm_res, ax=ax[2], legend=False)
plt.tight_layout()