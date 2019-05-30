# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:24:35 2019

@author: Wayner
"""

import numpy as np

def cholesky(A):
    m = len(A[:,0])
    n = len(A[0,:])
    R = A.copy()
    
    for i in range(m):
        for j in range(i+1,n):
            R[j,j:n] -= np.outer(R[i,j:n],R[i,j].conjugate()).transpose()[0]/R[i,i]
        R[i,i:m] /= np.sqrt(R[i,i])
        R[i+1:m,i] = np.array([0]*(m-i-1))
    return R
    
A = np.array([[2,-1,1,0],
              [-1,2,-1,1],
              [1,-1,2,-1],
              [0,1,-1,2]], dtype=float)

x = cholesky(A)