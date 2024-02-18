# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 21:45:53 2024

@author: Elève
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from PIL import Image


def rgb2gray(rgb):

    r, g, b = rgb[0], rgb[1], rgb[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


img = Image.open('gatto.jpg')
numpydata = np.asarray(img)


imagebw=np.array([[rgb2gray(rgb) for rgb in numpydata[i]] for i in range(len(numpydata))])

tipe=imagebw.shape
print(imagebw)
print(tipe)


def test(x,y):
    return (int(1200*x), int(1600*y))

print(test(0.6,0.4))

L=1
# parametres
def u0(x, L):
    ligne = int(x/1200)
    colonne = int((x-ligne)/1600)
    return imagebw[ligne][colonne]




K=1.
alpha=0
beta=0
theta=0

# discretisation temporelle
dt = 0.0001
T=0.5
N=int(T/dt)


# discretisation spatiale
n_x=50
n_y=50
dx=1/(n_x + 1)
dy=1/(n_y + 1)
X=np.linspace(0,1,n_x*n_y+1) #c'est l'abscisse... ici elle est double donc ca ne va pas coller

print(np.array([u0(x, L) for x in X[1:n_x*n_y]]))

# construction des matrices A et B et du vecteur c
coeff=K*dt/(dx**2)

c_x=dt/(dx*dx)
c_y=dt/(dy*dy)

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

M=Mat(c_x, c_y, n_x, n_y)
I=np.eye(n_x*n_y)



c=np.zeros(n_x*n_y)
c[0]=alpha*coeff #pour le moment c'est zéro
c[-1]=beta*coeff

print(len(M))
print(len(c))
print(len(I))

# initialisation
v=np.zeros((N+1,n_x*n_y+1))
v[0,1:n_x*n_y]=u0(X[1:n_x*n_y], L)
v[0,0]=alpha
v[0,n_x*n_y]=beta

# boucle en temps
for n in range(1,N+1):
    # resolution du systeme
    v[n,1:n_x*n_y]=np.linalg.solve(I, np.dot(M, v[n-1,1:n_x*n_y+1])+c)
    v[n,1:n_x*n_y+1]=np.dot(M,v[n-1,1:n_x*n_y+1])
    v[n,0]=alpha
    v[n,n_x*n_y]=beta

# affichage





fig=plt.figure(1)
plt.clf()
ax=fig.gca()
line, =ax.plot(X, v[0,:], 'r')
ax.axis([0, L, 0, 1])
ax.set_title('Donnee initiale')

# mise a jour de chaque plot
def runanimate(n):
    t=n*dt
    line.set_data(X, v[n,:])
    ax.set_title('Solution numerique a t=' + str('{:.3f}'.format(t)))
    
# lancement de l'animation
ani=anim.FuncAnimation(fig, runanimate, frames=np.arange(N+1), interval=50, repeat=False)

plt.show()

