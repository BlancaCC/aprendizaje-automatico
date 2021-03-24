# -*- coding: utf-8 -*-
"""
Exercise 2 
Author: Blanca Cano Camaro
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('_______LINEAR REGRESSION EXERCISE _______\n')
print('Exercise 1\n')
#input('\n Enter to start\n') UNCOMMENT


label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    '''quadratic error 
    INPUT
    x: input data matrix
    y: target vector
    w:  vector to 

    OUTPUT
    quadratic error >= 0
    '''
    
    y_hat = x.dot(w)

    error = np.square(y_hat - y.reshape(-1,1))
   
    return  error.mean()

# Gradiente Descendente Estocastico
def sgd():#?):
    #
    return w


def pseudoInverseMatrix ( X ):
    '''
    input: 
    X: is a matrix (must be a np.array) to use transpose and dot method
    return: hat matrix 
    '''

    #S =( X^TX ) ^{-1}
    simetric_inverse = np.linalg.inv( X.T.dot(X) )

    # S X^T = ( X^TX ) ^{-1} X^T
    return simetric_inverse.dot(X.T)


# Pseudoinversa	
def pseudoInverse(X, Y):
    ''' TO-DO matrix dimension is correct?
    input:
    X is  matrix, R^{m} \time R^{m} 
    Y is a vector (y_1, ..., y_m)
    
    OUTPUT: 
    w: weight vector
    '''
    X_pseudo_inverse = pseudoInverseMatrix ( X )
    Y_transposed = Y.reshape(-1, 1)
    
    w = X_pseudo_inverse.dot( Y_transposed)
    
    return w



# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

'''
w = sgd(?)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
'''


w_pseudoinverse = pseudoInverse(x, y) # change number
print('\nBondad del resultado para pseudo-inversa:')
print("  Ein:  ", Err(x, y, w_pseudoinverse))
print("  Eout: ", Err(x_test, y_test, w_pseudoinverse))
input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...
'''
print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign(?) 
'''
#Seguir haciendo el ejercicio...


#a = np.array([[1, 0], [0, 1]])


#def pseudoInverseGradientDescendent( eta, x, y ):
'''
def partialDerivativeE(j, x, y ):
    '#''
    input: 
    - j: vector number   
    - eta: learning rate   
    - x input data 
    - y target data 
    
    return partial derivative 
    '#''

    N = len (x)

    return 2.0/N * sum( [ np.dot(x[n][j] (H_projection(x[n])-y[n])) for n in range (N)] )


x = np.array([[1, 0], [0, 1]])
y = np.array([1,-1])
'''

