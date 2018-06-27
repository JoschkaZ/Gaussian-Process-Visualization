import numpy as np
import matplotlib.pyplot as plt

kernel = 2
def k(x,y):
    global kernel
    if kernel==1:
        return 1*x.T*y #linear
    elif kernel==2:
        return 1.*min(x,y)



#points to sample
x = np.arange(0,1,0.005)
n = len(x)

#covariance matrix
C = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        C[i,j] = k(x[i], x[j])
plt.figure(1)
plt.imshow(C, cmap='hot', interpolation='nearest')
plt.show()


for i in range(100):
    #sample from process
    r  = np.random.randn(n,1) #nx1 matrix of normal random numbers
    [u,s,vh] = np.linalg.svd(C) #factors C = np.dot(np.dot(u,np.diag(s)),vh)
    #s: vector(s) with singular values sorted in descending order
    #rows of vh are eigenvectors of C C^H

    z = np.dot(np.dot(u,np.sqrt(np.diag(s))),r)
    plt.figure(100)
    plt.plot(x,z)



plt.figure(100)
plt.show()
