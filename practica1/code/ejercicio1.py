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

def E(u,v):
    ''' (ESTO NO TIENE PORQUÉ SE  ASÍ)
    (vector, vector) -> real positivo 
    Error cuadrático medio
    '''

    
    return #function   

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return #Derivada parcial de E con respecto a u
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return #Derivada parcial de E con respecto a v

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

def gradient_descent(inicial_point, E, gradient_function,  eta=0.1):
    '''
    initicial point: w_0 
    E: error function 
    gradient_function
    eta:  step size 
    '''
    #### stop conditions   ######
    max_iter = 1000
    max_error =  0.0001
    ####

    iterations = 0
    current_error = E( intial_point[0], intial_point[1])
    w = initial_point

    while iterations < max_iter and error > max_error:

        w = w - eta * gradient_function(w[0], w[1])
        iterations += 1

    

    
    return w, iterations    


eta = 0.01 
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w, it = gradient_descent(?)


print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')


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

