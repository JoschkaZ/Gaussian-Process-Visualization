import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cholesky as chol


sig_f = 10.
l = 10.
sig_n = 1

def k_gaussian(x,y):
    global sig_f
    global l
    global sig_n
    return sig_f**2 * np.exp(-1./(2.*l**2) * (x-y)**2)

def k_noise(x,y):
    global sig_f
    global l
    global sig_n
    if x != y:
        return 0
    else:
        return sig_n**2


K_ff = np.zeros((100,100))
K_nn = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        K_ff[i,j] = k_gaussian(i,j)
        K_nn[i,j] = k_noise(i,j)

K_yy = K_ff + K_nn
print(K_yy.shape)
plt.figure(1)
plt.imshow(K_yy, cmap='hot', interpolation='nearest')
plt.show()

#draw
for i in range(10):
    f = np.dot(chol(K_yy),np.random.randn(100,1))

    plt.figure(2)
    plt.plot(f)
plt.show()
