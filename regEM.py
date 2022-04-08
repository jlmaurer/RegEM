#!/usr/env/bin python3

import os

import numpy as np


def center(x):
    # remove the mean from x
    xm = np.nanmean(x, axis=0)
    return x - np.tile(xm, (x.shape[0],1)), xm


def scale(x):
    # scale x by its std 
    s = np.nanstd(x, axis=0)
    return x / np.tile(s, (x.shape[0],1)), s


def standardize(x, factor=1):
    # return the standard z-score of x
    xc, xm = center(x) 
    xc, xs = scale(xc) 
    return xc, xm, xs / factor 


def rescale(X, C, D=None):
    # Scale X, C by variance (diag(C))
    if D is None:
        D = np.sqrt(np.diag(C))
    mask = np.abs(D) < 1e-6
    D[mask] = 1
    Xs = X / np.tile(D, (1, *X.shape[1:])) 
    Cs = C / np.tile(D, (1, *C.shape[1:]))
    return Xs, Cs, D



