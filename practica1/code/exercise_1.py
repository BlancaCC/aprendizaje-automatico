# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Autor Blanca Cano Camarero   
Grupo 2
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # to display 3d function 

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

## 1

def gradient_descent(initial_point, loss_function, gradient_function,  eta, max_iter, target_error):
    '''
    initicial point: w_0 
    E: error function 
    gradient_function
    eta:  step size 

    ### stop conditions ###
    max_iter
    target_error

    #### return ####
    (w,iterations)
    w: the coordenates that minimize E
    it: the numbers of iterations needed to obtain w
    
    '''

    iterations = 0
    error = E( initial_point[0], initial_point[1])
    w = initial_point
  
    while ( (iterations < max_iter) and(error > target_error)): 

        w = w - eta * gradient_function(w[0], w[1])
        
        iterations += 1
        error = loss_function(w[0], w[1])
 
    
    return w, iterations

############  2 ###################################
def E(u,v):
    '''
    Function to minimize
    '''
    return np.float64(
        ( u**3 * np.e**(v-2) - 2*v**2 * np.e**(-u) )**2
    )


def dEu(u,v):
    '''
    Partial derivate of E with respect to the variable u
    '''
    return np.float64(
        2
        *( u**3 * np.e**(v-2) - 2*v**2 * np.e**(-u))
        *( 3* u**2 * np.e**(v-2) + 2*v**2 * np.e**(-u))      
    )
    
def dEv(u,v):
    '''
    Partial derivate of E with respect to the variable v
    '''
    return np.float64(
        2
        *( u**3 * np.e**(v-2) - 2*v**2 * np.e**(-u) )
        *( u**3 * np.e**(v-2) - 4*v * np.e**(-u))
    )


def gradE(u,v):
    ''' 
        gradient of E
    '''
    return np.array([dEu(u,v), dEv(u,v)])



######  conditions  
eta = 0.1 
max_iter = 10000000000
target_error = 1e-14
initial_point = np.array([1.0,1.0])
w, it = gradient_descent( initial_point,E, gradE, eta, max_iter, target_error )


# DISPLAY FIGURE

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
plt.show()

######### Exercise 1, part 2 answers  ######
print('2 a) Function:  E(u,v) = (u^3 e^{(v-s)} - 2* v^2 e^{-u})^2')

print('dE_u = 2(u^3 e^{(v-s)} - 2* v^2 e^{-u})(3u^2e^{(v-2)} + 2 v^2 e^{-u} ), ')

print( 'dE_v =  2(u^3 e^{(v-s)} - 2* v^2 e^{-u})(u^3 e^{(v-2)} - 4 v e^{-u}')

print ('So de gradient is: \n nabla E(u,v) =(2(u^3 e^{(v-s)} - 2* v^2 e^{-u})(3u^2e^{(v-2)} + 2 v^2 e^{-u} ), 2(u^3 e^{(v-s)} - 2* v^2 e^{-u})(u^3 e^{(v-2)} - 4 v e^{-u}) )')

print ('2b) Numbers of iterations : ', it)
print ('2c) Final coodinates: (', w[0], ', ', w[1],')')



############################### EXERCISE 1 PART 3 #########

def f(x,y):
    '''
    Function to minimize
    '''
    return np.float64(
        (x+2)**2 + 2*(y-2)**2 + 2* np.sin( 2* np.pi * x)* np.sin( 2* np.pi * y)
    )

def dfx(x,y):
    '''
    Partial derivate of f with respect to the variable x
    '''
    return np.float64(
        2*( x+2 ) + 4* np.pi* np.cos(  2* np.pi * x )* np.sin(2* np.pi * y)
    )


def dfy(x,y):
    '''
    Partial derivate of f with respect to the variable y
    '''
    return np.float64(
        4*( y-2 ) + 4* np.pi* np.cos(  2* np.pi * y )* np.sin(2* np.pi * x)

    )


def gradF(x,y):
    ''' 
        gradient of E
    '''
    return np.array([dfx(x,y), dfy(x,y)])

######################## gradien descendent with trace  ############

def gradient_descent_trace(initial_point, loss_function, gradient_function,  eta, max_iter):
    '''
    initicial point: w_0 
    loss_function: error function 
    gradient_function
    eta:  step size 

    ### stop conditions ###
    max_iter

    #### return ####
    (w,iterations)
    w: the coordenates that minimize loss_function
    it: the numbers of iterations needed to obtain w
    
    '''

    iterations = 0
    error = loss_function( initial_point[0], initial_point[1])
    w = [initial_point]
  
    while iterations < max_iter: 

        new_w = w[-1] - eta * gradient_function(w[-1][0], w[-1][1])
        
        
        iterations += 1
        error = loss_function(new_w[0], new_w[1])
        w.append( new_w ) 
    
    return w, iterations



######  conditions  
smaller_eta = 0.01
bigger_eta = 0.1 
max_iter = 50
initial_point = np.array([-1.0,1.0])

#### run 
smaller_w, smaller_it = gradient_descent_trace( initial_point,f, gradF, smaller_eta, max_iter)
bigger_w, bigger_it = gradient_descent_trace( initial_point,f, gradF, bigger_eta, max_iter)

images_smaller_eta = [ f(x[0],x[1]) for x in smaller_w ]
images_bigger_eta = [ f(x[0],x[1]) for x in bigger_w]

print(f'With eta = {smaller_eta}, coordenates (x,y)= {(smaller_w[-1][0],smaller_w[-1][1])}, the number of iterations: {smaller_it} and the image is f(x,y) = {images_smaller_eta[-1]}')

print(f'With eta = {bigger_eta}, coordenates (x,y)={(bigger_w[-1][0],bigger_w[-1][1])}, the number of iterations: {bigger_it} and the image is f(x,y) = {images_bigger_eta[-1]}')


########### PLOTTING
x_label = 'Number of iterations'
y_label = 'f(x,y)'
bigger_eta_label = '$\eta$ = '+str(bigger_eta)
smaller_eta_label = '$\eta$ = '+str(smaller_eta)

## bigger eta ####
plt.clf()
plt.plot(images_bigger_eta, label=bigger_eta_label)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(f'Gradient descent of f with $\eta =$ {bigger_eta}')


plt.legend()
plt.show()

## smaller eta ###

plt.clf()
plt.plot(images_smaller_eta, label=smaller_eta_label)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(f'Gradient descent of f with $\eta =$ {smaller_eta}')


plt.legend()
plt.show()


## comparation ###
plt.clf()
plt.plot(images_bigger_eta, label=r"$\eta$ = 0.1")
plt.plot( images_smaller_eta, label=r"$\eta$ = 0.01")

plt.xlabel('Iterations')
plt.ylabel('f(x,y)')

plt.title("Comparation of the gradient descendent for $f$ changing eta value  ")


plt.legend()
plt.show()

### exta experimental ####

epsilon_eta = 1e-14
max_iter = 50
initial_point = np.array([-1.0,1.0])


epsilon_eta_label = '$\eta$ = '+str(epsilon_eta)
#### run 
epsilon_w, epsilon_it = gradient_descent_trace( initial_point,f, gradF, epsilon_eta, max_iter)
images_epsilon_eta = [ f(x[0],x[1]) for x in epsilon_w ]


print(f'With eta = {epsilon_eta}, coordenates (x,y)={(epsilon_w[-1][0],epsilon_w[-1][1])}, the number of iterations: {epsilon_it} and the image is f(x,y) = {images_epsilon_eta[-1]}')


plt.clf()
plt.plot(images_epsilon_eta, label=epsilon_eta_label)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(f'Gradient descent of f with $\eta =$ {epsilon_eta}')


plt.legend()
plt.show()


######################################################################

#### exercise 3.b

initial_points = map(np.array, [[0.1, 0.1], [1, 1], [-.5, -.5], [-1, -1]])
minimum_value = np.Infinity

print('{:^17}  {:^17}  {:^9}'.format('Initial', 'Final', 'Value'))

for initial in initial_points:
 
  w, _ = gradient_descent_trace(initial, f, gradF, 0.01, 50)
  local_minimum = w[-1]
  local_value = f(local_minimum[0], local_minimum[1])

  print('{}  {}  {: 1.5f}'.format(initial, local_minimum, local_value))

  




#input("\n--- Pulsar tecla para continuar ---\n")

