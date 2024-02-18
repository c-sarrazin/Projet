#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 2023

@author: courtes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


# parametres
def u0(x, L):
    return np.sin(np.pi*x/L)

K=1.
alpha=0
beta=0
theta=0

# discretisation temporelle
dt = 0.0001
T=0.5
N=int(T/dt)


# discretisation spatiale
L=1.
J=50
dx=L/J
X=np.linspace(0,L,J+1)

# construction des matrices A et B et du vecteur c
coeff=K*dt/(dx**2)
A=np.diag((1+2*theta*coeff)*np.ones(J-1), 0)+np.diag(-theta*coeff*np.ones(J-2), -1)+np.diag(-theta*coeff*np.ones( J-2), 1)
B=np.diag((1-2*(1-theta)*coeff)*np.ones(J-1), 0)+np.diag((1-theta)*coeff*np.ones(J-2), -1)+np.diag((1-theta)*coeff*np.ones(J-2), 1)

c=np.zeros(J-1)
c[0]=alpha*coeff
c[J-2]=beta*coeff

# initialisation
v=np.zeros((N+1,J+1))
v[0,1:J]=u0(X[1:J], L)
v[0,0]=alpha
v[0,J]=beta

# boucle en temps
for n in range(1,N+1):
    # resolution du systeme
    v[n,1:J]=np.linalg.solve(A, np.dot(B, v[n-1,1:J])+c)
    v[n,0]=alpha
    v[n,J]=beta

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
