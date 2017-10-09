import numpy as np
import scipy as sp
from SimPEG.Utils import mkvc

class spat_corr_field(object):  
    def  __init__(self,sigma = 4.4, corr = np.array([7,7,7],float),mu = 0, typ='gauss'):
     self.sigma = sigma
     self.corr  = corr
     self.mu    = mu
     self.typ   = typ
  
    # corr ..... vector of spatial correlation (variogram)
    # points ... list of x,y,z coordinates :type array, matrix [N x 3], x,y,z
    # sigma .... scalar, float
    # mu ....... mean value
    # typ ...... type of correlation
     
     
    def cov_matrix(self,points): 
     # Creates the covaraince matrix for set of points
     n     = len(points)
     C     = np.zeros((n,n))
     
     for i in range(n):
           point  = (points[i,:]) 
           if self.typ == 'gauss':
               x      = -0.5*np.square(points - np.tile(point,(n,1))).dot(1./self.corr)
           elif self.typ== 'exp':
               x      = -0.5*abs(points - np.tile(point,(n,1))).dot(1./self.corr)    
           C[:,i] = (self.sigma)*np.exp(x)  
           
     return C;
    
    def sort_points(self,points):
        n           = len(points)
        j           = 0
        points_sort = np.zeros((n,3))

        for i in range(n):            
            if points.shape[1] == 3:
               dist  = np.sqrt(points[:,0]**2 + points[:,1]**2 + points[:,2]**2 )   
            elif points.shape[1] == 2:
               dist  = np.sqrt(points[:,0]**2 + points[:,1]**2)
            ind   = sorted(range(len(dist)), key = lambda x:dist[x])
        
        for i in range(len(dist)):
            points_sort[j,:] = points[ind[i],:]
            j = j+1
            
        return points_sort             
     
    def svd_dcmp(self,points):  
        # Does decomposition of covariance matrix defined by set of points 
        # C ~= U*diag(s) * V, L = U*sqrt(s)
        C         = self.cov_matrix(points)
        U,ev,V    = np.linalg.svd(C)
        s         = np.sqrt(ev)
        L         = U.dot(sp.diag(s))
        
        return L,ev 
                        
    def values(self,points):  
        # Generates the actual field values, no mean yet
        # Field = mu + L*m, where m ~ iid from N(0,1)
        
        L,ev      = self.svd_dcmp(points)
        m         = np.random.normal(0,1,len(ev))
        
        return   L.dot(mkvc(m)) + self.mu
        
          
#=====================================================================
# Example:
"""
spf = spat_corr_field()
xyz = array([[1,1,1],[1,8,7],[4,3,7],[17,15,2],[3,23,8],[5,5,12]])
hx = 1*np.ones((16,))  # cell widths in the x-direction
hy = 1*np.ones((16,))  # cell widths in the y-direction 
hz = 1*np.ones((16,))  # cell widths in the z-direction 

mesh = Mesh.TensorMesh([hx,hy,hz]) 
points = mesh.GridCC

"""                         
                  
        