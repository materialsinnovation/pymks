import pymks

import numpy as np
import matplotlib.pyplot as plt

x0 = -10.
x1 = 10.
x = np.linspace(x0, x1, 1000)
def F(x):
    return np.exp(-abs(x)) * np.cos(2 * np.pi * x)
p = plt.plot(x, F(x), color='#1a9850')

import scipy.ndimage


n_space = 101
n_sample = 50
np.random.seed(201)
x = np.linspace(x0, x1, n_space)
X = np.random.random((n_sample, n_space))
y = np.array([scipy.ndimage.convolve(xx, F(x), mode='wrap') for xx in X])

from pymks import MKSLocalizationModel
from pymks import PrimitiveBasis

p_basis = PrimitiveBasis(n_states=2, domain=[0, 1])
model = MKSLocalizationModel(basis=p_basis)

model.fit(X, y)

y_pred = model.predict(X)

print(y_pred[0, :4])

print(model.coef_)
