# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:09:18 2019

@author: Wayner
"""

import numpy as np

def LU(A):
    m = len(A[:,0])
    n = len(A[0,:])
    U = A.copy()
    L = np.eye(m)
    P = np.eye(m)

    for j in range(n-1):
        absol = abs(U[j:m,j])
        l = max(absol)
        if l == 0:
            continue
        p = list(absol).index(l)
        
        if (p != 0):
            P[j], P[p+j] = P[p+j].copy(), P[j].copy()
            U[j], U[p+j] = U[p+j].copy(), U[j].copy()
            L[j,0:j], L[p+j,0:j] = L[p+j,0:j].copy(), L[j,0:j].copy()

        for i in range(j+1,m):
            L[i,j] = U[i,j]/U[j,j]
            if L[i,j] != 0:
                for k in range(j,n):
                    U[i,k] -= L[i,j]*U[j,k]
    
    return np.float64(L), np.float64(U), P

#A = np.array([[ 2,  1, 1,  0],
#       [ 4,  3,  3, 1],
#       [8,  7,  9, 5],
#       [ 6, 7, 9,  8]])


A = np.array([[1,3,5],
              [2,4,7],
              [1,1,0]])

A = np.float64(A)


l, u, p = LU(A)