 # -*- coding: utf-8 -*-
"""
Exercise 2 
Author: Blanca Cano Camaro
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
def STOP_EXECUTION_TO_SEE_RESULT():
        input('\n--- End of a section, press any enter to continue ---\n')

print('_______LINEAR REGRESSION EXERCISE _______\n')
print('Exercise 1')
input('\n Enter to start\n') 

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
def Error(x,y,w):
    '''quadratic error 
    INPUT
    x: input data matrix
    y: target vector
    w:  vector to 

    OUTPUT
    quadratic error >= 0
    '''
    error_times_n = np.linalg.norm(x.dot(w) - y.reshape(-1,1))**2
  
    return error_times_n/len(x)


def dError(x,y,w):
    ''' partial derivative
    '''

    return (2/len(x)*(x.T.dot(x.dot(w) - y.reshape(-1,1))))

# Gradiente Descendente Estocastico
def sgd(x,y, eta = 0.01, max_iter = 1000, batch_size = 32):
    '''
    Stochastic gradeint descent
    x: data set
    y: target vector
    eta: learning rate
    max_iter     
    '''

    w = np.zeros((x.shape[1], 1), np.float64)
    n_iterations = 0

    len_x = len(x)
    x_index = np.arange( len_x )
    batch_start = 0

    while n_iterations < max_iter:
            
            #shuffle and split the same into a sequence of mini-batches
        if batch_start == 0:
                x_index = np.random.permutation(x_index)
        iter_index = x_index[ batch_start : batch_start + batch_size]

        w = w - eta* dError(x[iter_index, :], y[iter_index], w)
       
        n_iterations += 1

        batch_start += batch_size
        if batch_start > len_x: # if end, restart
                batch_start = 0

    return w

def pseudoInverseMatrix ( X ):
    '''
    input: 
    X: is a matrix (must be a np.array) to use transpose and dot method
    return: hat matrix 
    '''

    '''
    #S =( X^TX ) ^{-1}
    simetric_inverse = np.linalg.inv( X.T.dot(X) )

    # S X^T = ( X^TX ) ^{-1} X^T
    return simetric_inverse.dot(X.T)
    '''
    return np.linalg.pinv(X)


# Pseudoinverse	
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


# Evaluating the autput

def performanceMeasurement(x,y,w):
    '''Evaluating the output binary case

    OUTPUT: 
    bad_negative, bad_positives, input_size
    '''

    sign_column = np.sign(x.dot(w)) - y.reshape(-1,1)

    bad_positives = 0
    bad_negatives = 0
    
    for sign in sign_column[:,0]:
        if sign > 0 :
                bad_positives += 1
        elif sign < 0 :
                bad_negatives += 1

    input_size = len(y)

    return bad_negatives, bad_positives, input_size

def evaluationMetrics (x,y,w, label = None):
    '''PRINT THE PERFORMANCE MEASUREMENT
    '''
    bad_negatives, bad_positives, input_size = performanceMeasurement(x,y,w)

    accuracy = ( input_size-(bad_negatives +bad_positives))*100 / input_size

    if label :
        print(label)
    print ( 'Input size: ', input_size )    
    print( 'Bad negatives :', bad_negatives)
    print( 'Bad positives :', bad_positives)
    print( 'Accuracy rate :', accuracy, '%')





    
## Draw de result

### scatter plot
def plotResults (x,y,w, title = None):
        label_5 = 1
        label_1 = -1

        labels = (label_5, label_1)
        colors = {label_5: 'b', label_1: 'r'}
        values = {label_5: 'Number 5', label_1: 'Number 1'}

        plt.clf()

        # data set plot 
        for number_label in labels:
                index = np.where(y == number_label)
                plt.scatter(x[index, 1], x[index, 2], c=colors[number_label], label=values[number_label])

        # regression line
        symmetry_for_cero_intensity = -w[0]/w[2]

        # en el caso de x1 = 1, tenemos 0 = w0 + w1 * w2 * x2
        # luego x2 = (-w0 - w1) /w2
        symmetry_for_one_intensity= (-w[0] - w[1])/w[2]
        plt.plot([0, 1], [symmetry_for_cero_intensity, symmetry_for_one_intensity], 'k-', label=(title+ ' regression'))

                

        if title :
                plt.title(title)
        plt.xlabel('Average intensity')
        plt.ylabel('Simmetry')
        plt.legend()
        plt.show()
        

### Draw a line ( it is a regression os a line so must have one line

### _____________ DATA ____________________

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

print("\n___ Goodness of the Stochastic Gradient Descendt (SGD) fit ___\n")

batch_sizes = [2,32, 200, 15000]
for _batch_size in batch_sizes:
        w = sgd(x,y, eta = 0.01, max_iter = 2000, batch_size = _batch_size)

        _title = f'SGD, batch size {_batch_size}'
        print( '\n\t'+_title)
        print ("Ein: ", Error(x,y,w))
        print ("Eout: ", Error(x_test, y_test, w))
        evaluationMetrics (x,y,w, '\nEvaluating output training data set')
        evaluationMetrics (x_test, y_test, w, '\nEvaluating output test data set')
        plotResults(x,y,w, title = _title)
        
        STOP_EXECUTION_TO_SEE_RESULT()


w_pseudoinverse = pseudoInverse(x, y) 
print("\n___ Goodness of the Pseudo-inverse fit ___\n")
print("  Ein:  ", Error(x, y, w_pseudoinverse))
print("  Eout: ", Error(x_test, y_test, w_pseudoinverse))

evaluationMetrics (x,y,w, '\nEvaluating output training data set')
evaluationMetrics (x_test, y_test, w, '\nEvaluating output test data set')
plotResults(x,y,w, title = 'Pseudo-inverse') 


STOP_EXECUTION_TO_SEE_RESULT()




print ('Exercise 2\n')




#### EXPERIMIENTS ###########

## a)
print('\nEXPERIMENT (a) \n')
def simula_unif(N, d, size):
        ''' generate a trining sample of N  points
in the square [-size,size]x[-size,size]
'''
        return np.random.uniform(-size,size,(N,d))


### data

size_training_example = 1000
dimension = 2
square_half_size = 1

training_sample = simula_unif( size_training_example,
                               dimension,
                               square_half_size)

STOP_EXECUTION_TO_SEE_RESULT()

plt.clf()
plt.scatter(training_sample[:, 0], training_sample[:, 1], c='b')
plt.title('Muestra de entrenamiento generada por una distribución uniforme')
plt.title('Training sample generated by a uniform distribution')
plt.xlabel('$x_1$ value')
plt.ylabel('$x_2$ value')

plt.show()

STOP_EXECUTION_TO_SEE_RESULT()

## b)

print('\nEXPERIMENT (b) \n')
def f(x1, x2):
	return np.sign(
            (x1 -0.2)**2
            +
            x2**2 -0.6
        ) 

#labels 
y = np.array( [f(x[0],x[1]) for x in training_sample ])

index = list(range(size_training_example))
np.random.shuffle(index)

percent_noisy_data = 10.0
size_noisy_data = int((size_training_example *percent_noisy_data)/ 100 )


noisy_y = np.copy(y)
for i in index[:size_noisy_data]:
    noisy_y[i] *= -1

    
## draw
labels = (1, -1)
colors = {1: 'blue', -1: 'red'}

plt.clf()

for l in labels:
	
	index = np.where(y == l)
	plt.scatter(training_sample[index, 0],
                    training_sample[index,1],
                    c=colors[l],
                    label=str(l))

plt.title('Labelled training sample before noise')
plt.xlabel('$x_1$ value')
plt.ylabel('$x_2$ value')
plt.legend()
plt.show()

STOP_EXECUTION_TO_SEE_RESULT()

plt.clf()

for l in labels:
	
	index = np.where(noisy_y == l)
	plt.scatter(training_sample[index, 0],
                    training_sample[index,1],
                    c=colors[l],
                    label=str(l))

plt.title('Labelled training sample after noise')
plt.xlabel('$x_1$ value')
plt.ylabel('$x_2$ value')
plt.legend()
plt.show()


STOP_EXECUTION_TO_SEE_RESULT()


#### C
print('\nEXPERIMENT (c) \n')
eta = 0.01
batch_size = 32
maximum_number_iterations = 1000


x = np.array( [
        np.array([ 1, x_n[0], x_n[1] ])
        for x_n in training_sample
])

w = sgd(x, noisy_y, eta, maximum_number_iterations, batch_size = 32)


_title = f'SGD, batch size {batch_size}'
print( '\n\t'+_title)
print ("Ein: ", Error(x,noisy_y,w))
evaluationMetrics (x,noisy_y,w, '\nEvaluating output training data set')

STOP_EXECUTION_TO_SEE_RESULT()

plt.clf()

for l in labels:
	
	index = np.where(noisy_y == l)
	plt.scatter(training_sample[index, 0],
                    training_sample[index,1],
                    c=colors[l],
                    label=str(l))

plt.title('Linear regression fit')
plt.xlabel('$x_1$ value')
plt.ylabel('$x_2$ value')

# regression line
y_0 = w[1]-w[0]
y_1 = -( w[1]-w[0])

plt.plot([-1, 1], [y_0, y_1], 'k-', label=('SGD regression'))

plt.show()

STOP_EXECUTION_TO_SEE_RESULT()

## d
print('\n EXPERIMENT (d), lineal regression\n')

def noisyVector(y,example_size, percent_noisy_data):
        '''
        y target vector to introduce noise
        size_training_example: number of point generated in each experiment,
        percent_noisy_data
        '''
        
        index = list(range(example_size))
        np.random.shuffle(index)
        
        size_noisy_data = int((example_size *percent_noisy_data)/ 100 )

        noisy_y = np.copy(y)
        for i in index[:size_noisy_data]:
                noisy_y[i] *= -1
        return noisy_y

        
def experiment(featureVector,
               number_of_repetitions = 1000,
               size_training_example = 1000,
               percent_noisy_data = 10.0
               ):
        '''
        INPUT
        featureVector: function that return  np.array 
        number_of_repetitions: experiment repetitions ,
        size_training_example: number of point generated in each experiment,
        percent_noisy_data: 

        OUTPUT
        (error_in, error_out)
        ''' 
        total_in_error = 0
        total_out_error = 0

        for _ in range( number_of_repetitions):
        ## data generation 
                training_sample = simula_unif( size_training_example,
                                               dimension,
                                               square_half_size)

                test_sample = simula_unif( size_training_example,
                                           dimension,
                                           square_half_size)
                test_y = np.array( [f(x[0],x[1]) for x in test_sample ])
                test_y = noisyVector(test_y, size_training_example, percent_noisy_data)
        
                y = np.array( [f(x[0],x[1]) for x in training_sample ])
                y = noisyVector(y, size_training_example, percent_noisy_data)
                #noise
                '''
                index = list(range(size_training_example))
                np.random.shuffle(index)
        
                size_noisy_data = int((size_training_example *percent_noisy_data)/ 100 )


                noisy_y = np.copy(y)
                for i in index[:size_noisy_data]:
                        noisy_y[i] *= -1
                '''
                # fit
                x = np.array( [
                        featureVector(x_n)
                        for x_n in training_sample
                ])

                x_test = np.array( [
                        featureVector(x_n)
                        for x_n in test_sample
                ])
                
                w = sgd(x, y, eta, maximum_number_iterations, batch_size = 32)

                total_in_error += Error(x,y,w)
                total_out_error += Error(x_test, test_y, w)
                
                
        error_in = float(total_in_error / number_of_repetitions)
        error_out = float(total_out_error / number_of_repetitions)

        return error_in, error_out



def linearFeatureVector(x_n):
        return np.array( [
                1,
                x_n[0],
                x_n[1]
               ] )
_number_of_repetitions = 1000
error_in, error_out = experiment( linearFeatureVector,
                                  number_of_repetitions = _number_of_repetitions,
                                  size_training_example = 1000,
                                  percent_noisy_data = 10.0
               )
print(f'The mean value of E_in in all {_number_of_repetitions} experiments is: {error_in}')
print(f'The mean value of E_out in all {_number_of_repetitions} experiments is: {error_out}')

STOP_EXECUTION_TO_SEE_RESULT()

# e)

print('\nEXPERIMENT (e)\n' )
eta = 0.01
batch_size = 32
maximum_number_iterations = 1000

def quadraticFeatureVector(x_n):
        return np.array([ 1,
                   x_n[0],
                   x_n[1],
                   x_n[0]*x_n[1],
                   x_n[0]* x_n[0],
                   x_n[1]* x_n[1]  ])

x = np.array( [
        quadraticFeatureVector(x_n)
        for x_n in training_sample
])

w = sgd(x, noisy_y, eta, maximum_number_iterations, batch_size = 32)

print('\nFor one experiment:')
_title = f'SGD, batch size {batch_size}'
print( '\n',_title)
print ("Ein: ", Error(x,noisy_y,w))
evaluationMetrics (x,noisy_y,w, '\nEvaluating output training data set')
print('')

STOP_EXECUTION_TO_SEE_RESULT()

## plotting

space = np.linspace(-1,1,300)

def equation (x,y,w):
        '''
        INPUT 
        x coordinate
        y coodinate 
        w weights vector
        
        OUTPUT
        Real number, the scalar product of features vector dot weights vector
        '''
        return ( w[0]
                 + w[1] * x
                 + w[2] * y
                 + w[3] * x * y
                 + w[4] * x**2
                 + w[5] * y**2
                )

error = 10**(-2.1)
x_image = []
y_image = []

last_value =  equation(-1,-1,w)
last_i = -1
last_j = -1
added = False
for i in space:
        for j in np.linspace(-1,0,150):
                actual_value = equation(i,j,w)
                if abs(last_value) < abs(actual_value) and abs(actual_value)<error :
                        if not added :
                                x_image.append(last_i)
                                y_image.append(last_j)
                                
                                added = True
                else:
                        
                        added = False
                last_j = j
                last_value = actual_value
        last_i = i
                
for i in space[::-1]:
        for j in np.linspace(0,1,150):
                actual_value = equation(i,j,w)
                if abs(last_value) < abs(actual_value) and abs(actual_value)<error :
                        if not added :
                                x_image.append(last_i)
                                y_image.append(last_j)
                                
                                added = True
                else:
                        
                        added = False
                last_j = j
                last_value = actual_value
        last_i = i

x_image.append(x_image[0])
y_image.append(y_image[0])
        
plt.clf()

for l in labels:
	
	index = np.where(noisy_y == l)
	plt.scatter(training_sample[index, 0],
                    training_sample[index,1],
                    c=colors[l],
                    label=str(l))

plt.title('Linear regression fit')
plt.xlabel('$x_1$ value')
plt.ylabel('$x_2$ value')
plt.plot(x_image,y_image, c = 'black', label='Regression model')
plt.legend( loc = 'lower left')
plt.show()

STOP_EXECUTION_TO_SEE_RESULT()

## EXPERIMENT
error_in, error_out = experiment( quadraticFeatureVector,
                                  number_of_repetitions = _number_of_repetitions,
                                  size_training_example = 1000,
                                  percent_noisy_data = 10.0
                                 )
print(f'The mean value of E_in in all {_number_of_repetitions} experiments is: {error_in}')
print(f'The mean value of E_out in all {_number_of_repetitions} experiments is: {error_out}')
print('==========================================')
STOP_EXECUTION_TO_SEE_RESULT()
