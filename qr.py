# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import numpy as np

def qr_GS(A):
    m = len(A[:,0])
    n = len(A[0,:])
    R = np.zeros([m,n])
    Q = np.zeros([m,n])
    v = np.zeros(m)
    for j in range(n):
        v = A[:,j].copy()
        for i in range(j):
            R[i,j] = np.dot(Q[:,i],A[:,j])
            v -= R[i,j]*Q[:,i]
        R[j,j] = np.linalg.norm(v)
        Q[:,j] = v/R[j,j]

    return Q.round(10), R.round(10)


def qr_MGS(A):
    m = len(A[:,0])
    n = len(A[0,:])
    R = np.zeros([m,n])
    Q = np.zeros([m,n])
    v = A.copy()
    for i in range(n):
        R[i,i] = np.linalg.norm(v[:,i])
        Q[:,i] = v[:,i]/R[i,i]
        for j in range(i+1,n):
            R[i,j] = np.dot(Q[:,i],v[:,j])
            v[:,j] -= R[i,j]*Q[:,i]
        
    return Q.round(10), R.round(10)


def qr_Hh(A):
    m = len(A[:,0])
    n = len(A[0,:])
    R = A.copy()
    v = np.zeros([n])
    for i in range(n):
        e1 = np.eye(1,m-i)
        x = np.array(R[i:m,i])
        v = np.sign(x[0])*np.linalg.norm(x)*e1 + x
        v /= np.linalg.norm(v)
        R[i:m,i:n] -= 2*np.outer(v, np.matmul(v, R[i:m,i:n]))
    
    R = R.ravel()[:n*n].reshape((n,n))
    Q = np.matmul(A, np.linalg.inv(R))
    
    return Q.round(10), R.round(10)   


def qr_givens(A):
    m = len(A[:,0])
    n = len(A[0,:])
    R = A.copy()
    Q = np.eye(m,m)
    for i in range(n):
        for j in range(i+1,m):
            a = R[i,i]
            b = R[j,i]
            if b:
                t = np.sqrt(a**2 + b**2)
                c = -a/t
                s = -b/t
                Q_transp = np.eye(m,m)
                Q_transp[i,i] = Q_transp[j,j] = c
                Q_transp[i,j] = s
                Q_transp[j,i] = -s
                R = np.matmul(Q_transp, R)
                Q = np.matmul(Q_transp, Q)
                
    return Q.transpose().round(10), R.round(10)



#A = np.array([[1,-1,0],[1,0,1],[-1,0,0],[1,1,0]], dtype=float)
A = np.array([[1,-1,0],[1,0,1],[-1,0,0]], dtype=float)
#A = np.array([[2,3],[5,4],[8,9]], dtype=float)

B_Hh = qr_Hh(A)
B_GS = qr_GS(A)
B_MGS = qr_MGS(A)
B_givens = qr_givens(A)
