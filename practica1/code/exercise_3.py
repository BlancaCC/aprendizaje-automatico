# -*- coding: utf-8 -*-
"""
Exercise 3 
Author: Blanca Cano Camaro
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
def STOP_EXECUTION_TO_SEE_RESULT():
        input('\n--- End of a section, press any enter to continue ---\n')

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


def gradf(x,y):
    ''' 
        gradient of E
    '''
    return np.array([dfx(x,y), dfy(x,y)])

def ddfxx(x,y):
    return 2 - 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def ddfyy(x,y):
   
    return 4 - 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def ddfxy(x,y):
    return 8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

def hessianf(x, y):
    return np.array([
        ddfxx(x,y),
        ddfxy(x,y),
        ddfxy(x,y),
        ddfyy(x,y),
    ]).reshape((2, 2))


def newton_trace(initial_point, fun, grad_fun, hessian, eta, max_iter):
    """ Newton method
    INPUT 
    - initial_point: 
    - f: differential function
    - grad_fun: Gradient
    - hessian: hessian
    - eta: learning rate
    - max_iter: number of iterations

    OUTPUT 
    w trace
    """

    w = initial_point
    w_list = [initial_point]
    iterations = 0

    while iterations < max_iter:
        w = w - eta *np.linalg.inv(hessian(w[0],w[1])).dot(grad_fun(w[0], w[1]))
        w_list.append(w)
        iterations += 1

    return np.array(w_list)


# Dicrease with the x-axis

## Exercise 1
 

######  conditions  
smaller_eta = 0.01
bigger_eta = 0.1 
max_iter = 50
initial_point = np.array([-1.0,1.0])

#### run 
smaller_w = newton_trace( initial_point,f, gradf, hessianf, smaller_eta, max_iter)
bigger_w = newton_trace( initial_point,f, gradf, hessianf, bigger_eta, max_iter)

images_smaller_eta = [ f(x[0],x[1]) for x in smaller_w ]
images_bigger_eta = [ f(x[0],x[1]) for x in bigger_w]

print(f'With eta = {smaller_eta}, coordenates (x,y)= {(smaller_w[-1][0],smaller_w[-1][1])}, the number of iterations: {max_iter} and the image is f(x,y) = {images_smaller_eta[-1]}')

print(f'With eta = {bigger_eta}, coordenates (x,y)={(bigger_w[-1][0],bigger_w[-1][1])}, the number of iterations: {max_iter} and the image is f(x,y) = {images_bigger_eta[-1]}')



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
plt.title(f"Newton's adjust of f with $\eta =$ {bigger_eta}")


plt.legend()
plt.show()

## smaller eta ###

plt.clf()
plt.plot(images_smaller_eta, label=smaller_eta_label)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(f"Newton's adjust of f with $\eta =$ {smaller_eta}")


plt.legend()
plt.show()


## comparation ###
plt.clf()
plt.plot(images_bigger_eta, label=r"$\eta$ = 0.1")
plt.plot( images_smaller_eta, label=r"$\eta$ = 0.01")

plt.xlabel('Iterations')
plt.ylabel('f(x,y)')

plt.title("Comparation of the newton's method for $f$ changing eta value  ")



plt.legend()
plt.show()


STOP_EXECUTION_TO_SEE_RESULT()

#### exercise 3.b
eta = 0.1
max_iter = 50
initial_points = map(np.array, [[-.5, -.5],[1, 1], [2.1, -2.1], [-3,3],  [-2, 2]])
minimum_value = np.Infinity

plt.clf()
print('{:^17}  {:^17}  {:^9}'.format('Initial', 'Final', 'Value'))

for initial in initial_points:
 
  w = newton_trace( initial,f, gradf, hessianf, eta, max_iter)
  local_minimum = w[-1]
  local_value = f(local_minimum[0], local_minimum[1])
  #plot
  images = [ f(x[0],x[1]) for x in w ]
  plt.plot(images, label=f'Initial point {initial}')
  

  print('{}  {}  {: 1.5f}'.format(initial, local_minimum, local_value))

plt.xlabel('Iterations')
plt.ylabel('f(x,y)')

plt.title("Comparation of the newton's method for $f$ changing initial point ")



plt.legend()
plt.show()


## without the biggest

initial_points = map(np.array, [[-.5, -.5],[1, 1], [-3,3],  [-2, 2]])
minimum_value = np.Infinity

plt.clf()
print('{:^17}  {:^17}  {:^9}'.format('Initial', 'Final', 'Value'))

for initial in initial_points:
 
  w = newton_trace( initial,f, gradf, hessianf, eta, max_iter)
  local_minimum = w[-1]
  local_value = f(local_minimum[0], local_minimum[1])
  #plot
  images = [ f(x[0],x[1]) for x in w ]
  plt.plot(images, label=f'Initial point {initial}')
  

  print('{}  {}  {: 1.5f}'.format(initial, local_minimum, local_value))

plt.xlabel('Iterations')
plt.ylabel('f(x,y)')

plt.title("Comparation of the newton's method for $f$ changing initial point  ")



plt.legend()
plt.show()
  
  
STOP_EXECUTION_TO_SEE_RESULT()

# let see gradient values 
points =  [[-0.38067878, -0.52778221],
           [1.06677195, 0.91078249],
           [ 3.26077803, -3.11750721] ,
           [-2.,  2.]]
             

for x,y in points :
    
    print(f'For {(x,y)}')
    print(f'\tThe gradient value is {gradf(x,y)}')
    print(f'\tThe inverse hessian is {np.linalg.inv(hessianf(x,y))}\n')

STOP_EXECUTION_TO_SEE_RESULT()
