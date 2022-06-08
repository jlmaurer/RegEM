#!/usr/env/bin python3
import os
import numpy as np


def regem(X, opt):
    '''
    Python implementation of the regularized 
    Expectation-Maximization algorithm for filling 
    in missing data
    '''
    ndata, nvar = X.shape

    # set up the options
    for key in opt_list:
        if key not in opt.keys():
            opt[key] = defaults[key]
    opt['ncv'] = ndata
    dofC = ndata - 1

    # Get the missing data mask
    nan_mask = np.isnan(X)
    if np.sum(nan_mask) == 0:
        print('No missing data')
        return
    
    # Compute missingness patterns
    np,kavlr,kmisr,prows,mp,iptrn = missingness_patterns(X)

    print('There are {} missingness patterns found.'.format(np))

    # Initial fill
    X, M = center(X)
    X[nan_mask] = 0
    C = np.dot(X.T, X) / dofC

    # X-validation
    if opt['truncslct'] == 'KCV':
        [incv, outcv, nin] = kcvindices(
            opts['ncv'], 
            opts['Kcv'], 
            ndata-opts['ncv']
        )

    # Iterate through the optimization
    B = []
    S = []
    Peff = []
    it = 0
    rdXmis = np.inf
    while (it < maxit) and (rdXmis > stagtol):
        it +=1
        CovRes = np.zeros(nvar,nvar)
        peff_ave = 0
        if True:
            [X, C, D] = rescale(X, C)

        for j in range(np):
            pm = length(kmisr[j])
            if pm > 0:
                pa = nvar - pm
            if pm < 1:
                print('No data at all in one or more rows of X')
            if opts['regress']=='mridge':
                b,s,_,peff = mridge(
                    C[kavlr[j],kavlr[j]],
                    C[kmisr[j],kmisr[j]], 
                    C[kavlr[j],kmisr[j]],
                    n - 1,
                    opts['optreg'],
                ) 
                B.append(b)
                Peff.append(peff)

            s_scaled = s * inflation * np.outer(D[kmisr[j]], D[kmisr[j]]) # not sure if this needs repmat'ed, looks like may need to be repeated pm times
            CovRes[kmisr[j],kmisr[j]] = CovRes[kmisr[j],kmisr[j]] + mp[j] * s_scaled
            S.append(s_scaled)
            Xmis[prows[j], kmisr[j]] = X[prows[j], kavlr[j]] * b

            # get error estimate
            if opt['truncslct'] == 'KCV':
                Xerr[prows[j], kmisr[j]] = repmat(x_rmserr * D[kmisr[j].T, mp[j], 1)
                




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


def nancov(x, flag=1):
    '''
    Returns the data covariance matrix ignoring nans
    '''
    xc,xm = center(x)
    m,n   = xc.shape
    
    # replace NaNs with zeros.
    xc[np.isnan(xc)] = 0.
      
    # normalization factor for each data
    nonnan  = np.ones(xc.shape)
    nonnan[np.isnan(xc)] = 0.
    nt = np.dot(nonnan.T, nonnan) - flag;  
      
    # set covariance to 0 when there are no terms to form sum
    nt[nt < 1] = 1;
    c = np.dot(xc.T,xc) / nt;
    return c, xm, xc

def peigs(A,rmax):
    '''Get the positive eigenvalues of a matrix'''

    # get the eigenvalues
    d,V = np.eig(A)

    # estimate number of positive eigenvalues of A
    d_min        = np.max(d) * np.max(A.shape) * 1e-8;
    r            = np.sum(d > d_min);

    # return only the positive e-values and vectors
    return d[:r], V[:, :r]


opt_list=[
    'regress', 
    'stagtol',
    'maxit',
    'inflation',
    'truncslct',
    'Kcv',
    'cv_norm',
    'ncv',
    'regpar', 
    'scalefac',
    'relvar_res',
    'Xmis0',
    'C0',
    'Xcmp',
    'neigs',
]

defaults = {
        'regress':'mridge',
        'stagtol':1e-2,
        'maxit':50,
        'inflation':1,
        'truncslct':'KCV',
        'Kcv':5,
        'cv_norm':2,
        'ncv':None,
        'regpar':None,
        'scalefac':None,
        'relvar_res':5e-2,
        'Xmis0':None,
        'C0':None,
        'Xcmp':None,
        'neigs':None,
}

