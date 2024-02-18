import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from PIL import Image

## encodage de l'image comme vecteur

img = Image.open('gatto.jpg')
numpydata = np.asarray(img)

#H, L = numpydata.shape[0],numpydata.shape[1]
# print(H, L)

def rgb2gray(rgb):

    r, g, b = rgb[0], rgb[1], rgb[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

# imagebw=np.array([[rgb2gray(rgb) for rgb in numpydata[i]] for i in range(len(numpydata))])
# numpy.core._exceptions._ArrayMemoryError: Unable to allocate 26.8 TiB for an array with shape (1920000, 1920000) and data type float64
# !!! 1600 par 1200 => matrices beaucoup trop grosses (haha il veut 26.8 TiB SOIT TRENTE MILLE GIGAS DE RAM ????)

imagebw = np.array([[255, 0, 255], [255, 0, 255], [255, 0, 255]])
L=3
H=3


 
## discrétisation spatiale ; LE MAILLAGE EST DONNÉ PAR LA PIXÉLISATION : carré de 1 par 1, coupé en 1600*1200
dx = 1/L
dy = 1/H

## discrétisation temporelle

T = 1
N = 50
dt = T/N

## initialisation

def vec2mat(m):
    return [mij for mi in m for mij in mi]

u_0 = vec2mat(np.transpose(imagebw)) # pour ranger par colonnes comme allaire (décision de con)
u_t = [u_0]+N*[len(u_0)*[0.]]



## construction matrice

def Mat(c_x, c_y, n_x, n_y):
    D = np.diag([-c_y]*(n_y-1), k=1)+ np.diag([-c_y]*(n_y-1), k=-1) + np.diag([1+2*(c_x+c_y)]*n_y, k=0)
    O = np.zeros((n_y, n_y))
    E = np.diag([-c_y]*(n_y), k=0)

    def Mat(i,j):
        if i == j:
            return D
        elif abs(i-j) == 1:
            return E
        else:
            return O
    
    return np.block([[Mat(i,j) for j in range(n_x)] for i in range(n_x)])

c_x=dt/(dx*dx)
c_y=dt/(dy*dy)

M=Mat(c_x, c_y, L, H)


for n in range(1,N+1):
    u_t[n]=np.linalg.solve(M, u_t[n-1])
    # ou np.dot(M, u_t[n-1]) ??
    # quid des conditions au bords ? 

def mat2vec(v, l, h):
    assert len(v) == l*h
    return [[v[h*i+j] for j in range(h)] for i in range(l)]

print(np.transpose(mat2vec(u_t[1], L, H)))