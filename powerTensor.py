import numpy as np
from numba import njit
import healpy as hp
import math

@njit
def jacobi(A):
    n     = A.shape[0]            # matrix size #columns = #lines
    maxit = 100                   # maximum number of iterations
    eps   = 1.0e-15               # accuracy goal
    pi    = np.pi        
    info  = 0                     # return flag
    ev    = np.zeros(n,float)     # initialize eigenvalues
    U     = np.zeros((n,n),float) # initialize eigenvector
    for i in range(0,n): U[i,i] = 1.0

    for t in range(0,maxit):
        s = 0;    # compute sum of off-diagonal elements in A(i,j)
        for i in range(0,n): s = s + np.sum(np.abs(A[i,(i+1):n]))
        if (s < eps): # diagonal form reached
            info = t
            for i in range(0,n):ev[i] = A[i,i]
            break
        else:
            limit = s/(n*(n-1)/2.0)       # average value of off-diagonal elements
            for i in range(0,n-1):       # loop over lines of matrix
                 for j in range(i+1,n):  # loop over columns of matrix
                    if (np.abs(A[i,j]) > limit):      # determine (ij) such that |A(i,j)| larger than average 
                                                      # value of off-diagonal elements
                        denom = A[i,i] - A[j,j]       # denominator of Eq. (3.61)
                        if (np.abs(denom) < eps): phi = pi/4         # Eq. (3.62)
                        else: phi = 0.5*np.arctan(2.0*A[i,j]/denom)  # Eq. (3.61)
                        si = np.sin(phi)
                        co = np.cos(phi)
                        for k in range(i+1,j):
                            store  = A[i,k]
                            A[i,k] = A[i,k]*co + A[k,j]*si  # Eq. (3.56) 
                            A[k,j] = A[k,j]*co - store *si  # Eq. (3.57) 
                        for k in range(j+1,n):
                            store  = A[i,k]
                            A[i,k] = A[i,k]*co + A[j,k]*si  # Eq. (3.56) 
                            A[j,k] = A[j,k]*co - store *si  # Eq. (3.57) 
                        for k in range(0,i):
                            store  = A[k,i]
                            A[k,i] = A[k,i]*co + A[k,j]*si
                            A[k,j] = A[k,j]*co - store *si
                        store = A[i,i]
                        A[i,i] = A[i,i]*co*co + 2.0*A[i,j]*co*si +A[j,j]*si*si  # Eq. (3.58)
                        A[j,j] = A[j,j]*co*co - 2.0*A[i,j]*co*si +store *si*si  # Eq. (3.59)
                        A[i,j] = 0.0                                            # Eq. (3.60)
                        for k in range(0,n):
                             store  = U[k,j]
                             U[k,j] = U[k,j]*co - U[k,i]*si  # Eq. (3.66)
                             U[k,i] = U[k,i]*co + store *si  # Eq. (3.67)
        info = -t # in case no convergence is reached set info to a negative value "-t"
    return ev,U,t


def ang_mom_mat(l):
    j=np.zeros((3, int(2*l+1), int(2*l+1)), dtype=np.complex128)
    i = np.arange(0, int(2*l))
    k = i+1
    d = np.arange(int(2*l+1))
    j[0, i, k] = j[0, k, i] = np.sqrt(l*(l+1)-(l-k)*(l-k+1))/2
    j[1, i, k] = np.sqrt(l*(l+1)-(l-k)*(l-k+1))/2j
    j[1, k, i] = np.conj(j[1, i, k])
    j[2, d, d] = np.linspace(l, -l, int(2*l+1))
    return j

#def __alm(l,alm,lmax):
#    a = np.zeros((2*l+1), dtype=np.complex128)
#    a[range(l,2*l+1)] = alm[hp.Alm.getidx(lmax,l,np.arange(l+1))]
#    a[range(l-1,-1,-1)] = np.resize([-1,1], l)*np.conj(a[range(l+1,2*l+1)])
#    return a[::-1]

def __alm(l,al,lmax):
    a=np.zeros((2*l+1))+0j
    a[range(0, l+1)]=al[hp.Alm.getidx(lmax,l,np.arange(l, -1, -1))]
    a[range(l+1, 2*l+1)]=np.resize([-1,1], l)*np.conj(a[range(l-1, -1, -1)])
    return a

def __PEV(alm,lmax, l, pevType, eigMethod):
    jk = ang_mom_mat(l)
    alm_ = __alm(l,alm,lmax)
    alm_c = np.conj(alm_)
    A = np.zeros((3,3),np.complex128)
    for i in range(0,3):
        for k in range(0,3):
            A[i,k] = np.linalg.multi_dot([alm_c,jk[i],jk[k],alm_])/(l*(l+1)*(2*l+1))

    if eigMethod == "numpy_linagl_eig":
        eig_val,eig_vec = np.linalg.eig(np.real(A))
    if eigMethod == "jacobi":
        eig_val,eig_vec, _ = jacobi(np.real(A))
    arg_sort = np.argsort(eig_val)
    if pevType=="max":
        pev = eig_vec[:,arg_sort[2]]
    elif pevType=="min":
        pev = eig_vec[:,arg_sort[0]]
    elif pevType=="middle":
        pev = eig_vec[:,arg_sort[1]]
    elif pevType=="all":
        pev = eig_vec
    else:
        raise ValueError(f"'{pevType}' is not a valid value for pevType; supported values are 'min', 'max', 'middle' and 'all'")
    return pev, eig_val

def PEV(alm, l, eig_value=False, pevType="max", eigMethod="numpy_linagl_eig"):
    lmax=hp.Alm.getlmax(alm.size)
    if type(l) == int:
        assert 2<=l<=lmax, 'l must follow this condition "2 <= l <= lmax"'
        pev, eig_val = __PEV(alm, lmax, l, pevType)
    elif type(l) == np.ndarray or type(range(2)) == range or type(l)==list or type(l)==tuple:
        assert 2<=l[0] and l[~0]<=lmax, 'l must follow this condition "2 <= l <= lmax"'
        eig_val = np.zeros((len(l),3))
        pev = np.zeros((len(l), 3, 3)) if pevType=="all" else np.zeros((len(l), 3))
        for i, ll in enumerate(l):
            pev[i], eig_val[i] = __PEV(alm, lmax, ll, pevType, eigMethod)
    else:
        raise TypeError("l must be a integer or list/array of integers.")
    if eig_value==True:
        return pev, eig_val
    return pev

def alignmentTensor(PEVs, eigMethod, eig_value=False):
    if PEVs.ndim == 1:
        PEVs_ = PEVs.reshape(1, 3, 1)
    else:
        PEVs_ = PEVs.reshape(len(PEVs), 3, 1)
    PEVs_t = np.transpose(PEVs_, axes=(0, 2, 1))
    AT_ = np.matmul(PEVs_, PEVs_t)
    AT = np.mean(AT_, axis=0) 
    if eigMethod == "numpy_linagl_eig":
        AT_eig_val, AT_eig_vec = np.linalg.eig(AT)
    if eigMethod == "jacobi":
        AT_eig_val, AT_eig_vec, _ = jacobi(AT)
    AT_PEV = AT_eig_vec[:, np.argmax(AT_eig_val)]
    if eig_value == True:
        return AT_PEV, AT_eig_val
    return AT_PEV

# To calculate Power Entropy
def power_entropy(l,ev):
    sum_lam=ev[0:l-1]/ev[0:l-1].sum(axis=1,keepdims=True)
    p_entropy=(-sum_lam*np.log(sum_lam)).sum(axis=1)
    return p_entropy

@njit
def ang_btw_mpl(l,pev):
    ang = np.zeros((l-1,l-1))
    for i in range(l-1):
        for j in range(i):
            ang[i,j] = np.rad2deg(np.arccos(np.abs(np.sum(pev[i]*pev[j]))))
    return ang

@njit
def ang_btw_axis(l,pev, vec):
    ang = np.zeros((l-1))
    for i in range(l-1):
        ang[i] = np.rad2deg(np.arccos(np.abs(np.sum(pev[i]*vec))))
    return ang

# To calculate Comulative Binomial Probabilty
def cbp(k,n,p = 0.05):
    c = 0
    for i in range(k):
        c += math.comb(n,i)*(p**i)*(1-p)**(n-i)
    return 1-c

def vec2ang(PEVs, unit="Radian"):
    phi_d,theta_d = hp.vec2ang(PEVs,lonlat=True)

    for i in range(len(phi_d)):
        if phi_d[i]>180:
            phi_d[i] = phi_d[i]-360
            
    if unit=="degree" or unit=="deg":
        return theta_d, -phi_d
    else:
        return np.deg2rad(theta_d), -np.deg2rad(phi_d)
