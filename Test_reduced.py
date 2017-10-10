# Testing the COV matrices estimates for reduced subset of points
import random
import numpy as np
import scipy as sp
from SimPEG import Mesh, Utils 
from SimPEG.Utils import mkvc
from spat_corr_field import spat_corr_field
from operator import itemgetter
import matplotlib.pyplot as plt

# =========== Grid of points: =================================================
hx = 1*np.ones((20,))  # cell widths in the x-direction
hy = 1*np.ones((20,))  # cell widths in the y-direction 
hz = 1*np.ones((16,))  # cell widths in the y-direction

mesh   = Mesh.TensorMesh([hx,hy,hz]) 
points = mesh.gridCC
n      = len(points)

# ========== Setup the full C and field ========================================
sigma     = 7
corr      = np.array([5,2,4.3])
mu        = 0;
pole      = spat_corr_field(sigma, corr, mu,'gauss')
C         = pole.cov_matrix(points)
F         = pole.values(points)
f3d       = np.reshape(F,((mesh.nCx,mesh.nCy, mesh.nCz)),order='F')

# =========== Subset of points =================================================
N           = 100; # number of points
subset      = np.zeros((N,3))

for i in range(N):
    subset[i,:] = [random.random()*len(hx),random.random()*len(hy), random.random()*len(hz)]
subset_sort = pole.sort_points(subset)

C_red     = pole.cov_matrix(subset_sort)
Lr, evr   = pole.svd_dcmp(subset_sort)

K         = 1000;
prum      = 0
cov       = np.zeros((len(subset),len(subset)));
for i in range(K):
     m    = np.random.normal(0,1,len(evr))
     f    = Lr.dot(mkvc(m))   # field based on reduced cov matrix
     cov  = cov + (f)*mkvc(f,2)  # estimate of cov matrix based on the subset points
     prum = prum + f.mean()  #average of the field
     
C_est     = (1./K)*cov  # the "averaged" covariance matrix based on points realizations
prum_est  = (1./K)*prum 

print ('Estimated mean, estimated sigma')




# =========== Some visualization ===============================================
fig = plt.figure(figsize=plt.figaspect(0.25))
ax  = fig.add_subplot(1, 5, 1)
ax.set_title('Full covariance',size = 9)
plt.imshow(C)
plt.colorbar(orientation='horizontal')
ax  = fig.add_subplot(1, 5, 2)
ax.set_title('The field on full grid',size = 9)
plt.imshow(f3d[:,:,6])
plt.colorbar(orientation='horizontal')
ax  = fig.add_subplot(1, 5,3)
ax.set_title('One of the realizations',size = 9)
plt.scatter(subset[:,0],subset[:,1],s = subset[:,2], c=f)
plt.colorbar(orientation='horizontal')
ax  = fig.add_subplot(1, 5,4)
ax.set_title('The reduced covariance',size = 9)
plt.imshow(C_red)
plt.colorbar(orientation='horizontal')
ax  = fig.add_subplot(1, 5,5)
ax.set_title('The difference:C_red - C_est',size = 9)
plt.imshow(C_est - C_red)
plt.colorbar(orientation='horizontal')
plt.show()



