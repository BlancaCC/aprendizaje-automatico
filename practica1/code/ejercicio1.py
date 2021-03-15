# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Autor Blanca Cano Camarero   
Grupo 2
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

## 1

def gradient_descent(initial_point, E, gradient_function,  eta, max_iter, max_error):
    '''
    initicial point: w_0 
    E: error function 
    gradient_function
    eta:  step size 

    ### stop conditions 
    max_iter
    max_error
    '''

    iterations = 0
    error = E( initial_point[0], initial_point[1])
    w = initial_point
  
    while ( (iterations < max_iter) and(error > max_error)): 

        w = w - eta * gradient_function(w[0], w[1])
        
        iterations += 1
        error = E(w[0], w[1])
        print(w)
 
    
    return w, iterations

## 2
def E(u,v):
    '''
    Function to minimize
    '''
    return np.float64(
        (
            u**3 * np.e**(v-2) - 2*v**2 * np.e**(-u)
        )**2
    )


def dEu(u,v):
    '''
    Partial derivate of E with respect to the variable u
    '''
    return np.float64(
        2*(
           3* u**2 * np.e**(v-2) + 2*v**2 * np.e**(-u)
        )
    )
    
def dEv(u,v):
    '''
    Partial derivate of E with respect to the variable v
    '''
    return np.float64(
        2*(
        u**3 * np.e**(v-2) - 4*v * np.e**(-u)
        )
    )


def gradE(u,v):
    ''' 
        gradient of E
    '''
    return np.array([dEu(u,v), dEv(u,v)])

eta = 0.01 
max_iter = 100#00000000
error_to_get = 1e-14
initial_point = np.array([1.0,1.0])
w, it = gradient_descent( initial_point,E, gradE, eta, max_iter, error_to_get )


print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

'''
# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')

input("\n--- Pulsar tecla para continuar ---\n")

'''
