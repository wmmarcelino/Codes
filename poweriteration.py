#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:15:59 2019

@author: wayner
"""

import numpy as np

def poweriteration (A, v, n):
    vk = v.copy()
    for k in range(1,n+1):
        w = np.matmul(A,vk)
        vk = w/np.linalg.norm(w,2)
        l = np.inner(np.matmul(A,vk), vk)
    return vk, l

def invpoweriteration (A, v, n, sigma):
    vk = v.copy()
    I = np.eye(len(A))
    for k in range(1,n+1):
        w = np.matmul(np.linalg.inv(A-sigma*I),vk)
        vk = w/np.linalg.norm(w,2)
        l = np.inner(np.matmul(A,vk),vk)
    return vk, l

A = np.array([[1,1,0],
              [0,1,1],
              [1,0,1]])
    
v = np.array ([2,1,1])

#avetor, avalor = poweriteration (A,v,10)

avetor, avalor = invpoweriteration (A,v,20,1.7)