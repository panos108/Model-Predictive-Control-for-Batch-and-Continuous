import numpy as np
from train_ann import *

x_m = 1000  # number of samples
x = np.random.default_rng().uniform(-5, 5, x_m)
y = np.random.default_rng().uniform(-5, 5, x_m)
X = np.array((x, y)).T

F = []
for i in range(0, x_m, 1):
    f = np.array([[100 * (y[i] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2, x[i]*y[i]]])
    F.append(*f.reshape(1,2))
F = np.array(F)
if f.ndim==1:
    F = F.reshape(-1, 1)

model = Model(X.shape[1],F.shape[1])
ANN = train_ann(model, X, F, learning_rate=1e-3, plot=False)


ys = ANN.predict(X)


print('2')
