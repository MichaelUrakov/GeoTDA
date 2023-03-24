import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from gtda.images import *
from gtda.homology import VietorisRipsPersistence, EuclideanCechPersistence
from gtda.utils import *

fig, ax = plt.subplots(1, 2, figsize=(18, 6))

np.random.seed(0)

data = np.random.randn(10, 2)[None, :, :]

ax[0].set_title('Pointcloud', fontsize=16)
ax[0].scatter(x=data[0][:, 0], y=data[0][:, 1], alpha=0.5)


# data = Binarizer().fit_transform(data)


data_pers = EuclideanCechPersistence(homology_dimensions=(0, 1, 2)).fit_transform(data)


data_pers = np.array([(i[2], (i[0], i[1])) for i in data_pers[0]], dtype=[('dim', int), ('lifetime', tuple)])

gd.plot_persistence_barcode(data_pers, axes=ax[1], legend=True)
ax[1].set_title('Its Persistence barcode', fontsize=16)