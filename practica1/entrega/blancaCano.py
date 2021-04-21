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
def STOP_EXECUTION_TO_SEE_RESULT():
        input('\n--- End of a section, press any enter to continue ---\n')
        


print('GRADIENT DESCENDENT\n')
print('Exercise 1\n')

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
STOP_EXECUTION_TO_SEE_RESULT()
## smaller eta ###

plt.clf()
plt.plot(images_smaller_eta, label=smaller_eta_label)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(f'Gradient descent of f with $\eta =$ {smaller_eta}')


plt.legend()
plt.show()

STOP_EXECUTION_TO_SEE_RESULT()
## comparation ###
plt.clf()
plt.plot(images_bigger_eta, label=r"$\eta$ = 0.1")
plt.plot( images_smaller_eta, label=r"$\eta$ = 0.01")

plt.xlabel('Iterations')
plt.ylabel('f(x,y)')

plt.title("Comparation of the gradient descendent for $f$ changing eta value  ")


plt.legend()
plt.show()
STOP_EXECUTION_TO_SEE_RESULT()



### exta experimental ####

epsilon_eta = 1e-14
max_iter = 50
initial_point = np.array([-1.0,1.0])


epsilon_eta_label = '$\eta$ = '+str(epsilon_eta)
#### run 
epsilon_w, epsilon_it = gradient_descent_trace( initial_point,f, gradF, epsilon_eta, max_iter)
images_epsilon_eta = [ f(x[0],x[1]) for x in epsilon_w ]


print(f'With eta = {epsilon_eta}, coordenates (x,y)={(epsilon_w[-1][0],epsilon_w[-1][1])}, the number of iterations: {epsilon_it} and the image is f(x,y) = {images_epsilon_eta[-1]}')
STOP_EXECUTION_TO_SEE_RESULT()

plt.clf()
plt.plot(images_epsilon_eta, label=epsilon_eta_label)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(f'Gradient descent of f with $\eta =$ {epsilon_eta}')


plt.legend()
plt.show()

STOP_EXECUTION_TO_SEE_RESULT()
## smaller eta and tiny one ###

plt.clf()

plt.plot(images_epsilon_eta, label=epsilon_eta_label)
plt.plot( images_smaller_eta, label=r"$\eta$ = 0.01")

plt.xlabel('Iterations')
plt.ylabel('f(x,y)')

plt.title("Comparation of the gradient descendent for $f$ changing eta value  ")


plt.legend()
plt.show()
STOP_EXECUTION_TO_SEE_RESULT()
######################################################################

#### exercise 3.b

initial_points = map(np.array, [[-.5, -.5],[1, 1], [2.1, -2.1], [-3,3],  [-2, 2]])
minimum_value = np.Infinity

print('{:^17}  {:^17}  {:^9}'.format('Initial', 'Final', 'Value'))

for initial in initial_points:
 
  w, _ = gradient_descent_trace(initial, f, gradF, 0.01, 50)
  local_minimum = w[-1]
  local_value = f(local_minimum[0], local_minimum[1])

  print('{}  {}  {: 1.5f}'.format(initial, local_minimum, local_value))

  
STOP_EXECUTION_TO_SEE_RESULT()


"""
Exercise 2 
Author: Blanca Cano Camarero
"""

def STOP_EXECUTION_TO_SEE_RESULT():
        input('\n--- End of a section, press any enter to continue ---\n')
        

print('_______LINEAR REGRESSION EXERCISE _______\n')
print('Exercise 1')
input('\n Enter to start\n') 

label5 = 1
label1 = -1


def readData(file_x, file_y):
        '''
        function for read data
        '''
	# reads files
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


def Error(x,y,w):
    '''quadratic error 
    INPUT
    x: input data matrix
    y: target vector
    w:  vector to 

    OUTPUT
    quadratic error >= 0
    '''
    error_times_n = np.float64(np.linalg.norm(x.dot(w) - y.reshape(-1,1))**2)
  
    return np.float64(error_times_n/len(x))


def dError(x,y,w):
    ''' gradient
    OUTPUT
    column vector
    '''
    return  2/len(x)*(x.T.dot(x.dot(w) - y.reshape(-1,1)))
    


def sgd(x,y, eta = 0.01, max_iter = 1000, batch_size = 32, error=10**(-10)):
        '''
        Stochastic gradient descent
        INPUT 
        x: data set
        y: target vector
        eta: learning rate
        max_iter 

        OUTPUT 
        w: weight vector
        '''
  
        #initialize data
        w = np.zeros((x.shape[1], 1), np.float64)
        n_iterations = 0

        len_x = len(x)
        x_index = np.arange( len_x )
        batch_start = 0
        w_error = Error(x,y,w)

        while n_iterations < max_iter and w_error > error :
  
                #shuffle and split the same into a sequence of mini-batches
                np.random.shuffle(x_index)
                for batch_start in range(0,  len_x, batch_size):
                        iter_index = x_index[ batch_start : batch_start + batch_size]

        
                        w = w - eta* dError(x[iter_index, :], y[iter_index], w)
        
                n_iterations += 1
                w_error = Error(x,y,w)

   
        return w


def sgd_exact_number_iter(x,y, eta = 0.01, max_iter = 1000, batch_size = 32, error = 10**(-10)):
        '''
        Stochastic gradient descent
        INPUT 
        x: data set
        y: target vector
        eta: learning rate
        max_iter 
        OUTPUT 
        w: weight vector
        '''
        #initialize data
        w = np.zeros((x.shape[1], 1), np.float64)
    
        n_iterations = 0
        batch_start = 0
        len_x = len(x)
    
        x_index = np.arange( len_x )
        w_error = Error(x,y,w)
 
        while n_iterations < max_iter and w_error > error:
                #shuffle and split the same into a sequence of mini-batches
                if batch_start == 0:
                        x_index = np.random.permutation(x_index)
                iter_index = x_index[ batch_start : batch_start + batch_size]

                w = w - eta* dError(x[iter_index, :], y[iter_index], w)
                
                n_iterations += 1

                batch_start += batch_size
                if batch_start >= len_x: # if end, restart
                        batch_start = 0
                
                w_error = Error(x,y,w)


        return w

def pseudoInverseMatrix ( X ):
    '''
    INPUT 
    X: is a matrix (must be a np.array) to use transpose and dot method
    OUTPUT
    hat matrix 
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
    ''' 
    INPUT
    X is the feature matrix 
    Y is the target vector (y_1, ..., y_m)
    
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

    INPUT
    X is the feature matrix 
    Y is the target vector (y_1, ..., y_m)
    
    OUTPUT: 
    w: weight vector
    OUTPUT: 
    bad_negative, bad_positives, input_size
    '''

    # defference between the sign of the regression and the target vector
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
    print (f'For w^T = {w.reshape(1,-1)}')
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
        # x = 0
        symmetry_for_cero_intensity = -w[0]/w[2]

        #  x = 1, 0 = w0 + w1 * w2 * x2
        # then y = (-w0 - w1) /w2
        symmetry_for_one_intensity= (-w[0] - w[1])/w[2]

        #plotting order
        plt.plot([0, 1], [symmetry_for_cero_intensity, symmetry_for_one_intensity], 'k-', label=(title+ ' regression'))

                

        if title :
                plt.title(title)
        plt.xlabel('Average intensity')
        plt.ylabel('Simmetry')
        plt.legend()
        plt.show()
        

def plotResultMultiplesLines(x,y,multiple_w, main_title = None, multiples_title = None):
        '''
        INPUT 
        x featue matrx
        y labels vector 
        multiple_w vector of different weight vector
        multiple_titles
        '''
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

        for i in range(len(multiple_w)):
                w = multiple_w[i]
                title = multiple_title[i]
                # regression line
                # x = 0
                symmetry_for_cero_intensity = -w[0]/w[2]

                #  x = 1, 0 = w0 + w1 * w2 * x2
                # then y = (-w0 - w1) /w2
                symmetry_for_one_intensity= (-w[0] - w[1])/w[2]

                #plotting order
                plt.plot([0, 1],
                         [symmetry_for_cero_intensity, symmetry_for_one_intensity],
                         #'k-',
                         label=(title+ ' regression'))

                

        if main_title :
                plt.title(main_title)
        plt.xlabel('Average intensity')
        plt.ylabel('Simmetry')
        plt.legend()
        plt.show()
        

### _____________ DATA ____________________

# Reading training data set 
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Reading test data set 
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

w_pseudoinverse = pseudoInverse(x, y) 
print("\n___ Goodness of the Pseudo-inverse fit ___\n")
print("  Ein:  ", Error(x, y, w_pseudoinverse))
print("  Eout: ", Error(x_test, y_test, w_pseudoinverse))

evaluationMetrics (x,y,w_pseudoinverse, '\nEvaluating output training data set')
evaluationMetrics (x_test, y_test, w_pseudoinverse, '\nEvaluating output test data set')
plotResults(x,y,w_pseudoinverse, title = 'Pseudo-inverse') 

print(f'\nThe weight vector is {w_pseudoinverse}')

STOP_EXECUTION_TO_SEE_RESULT()


print("\n___ Goodness of the Stochastic Gradient Descendt (SGD) fit ___\n")

batch_sizes =[1,32,200,len(y)] #batch sizes compared in the experiment

n_iterations = [50,300] 
for iteration in n_iterations:
        multiple_w = [w_pseudoinverse]
        multiple_title = ['pseudo-inverse']

        for _batch_size in batch_sizes:
                w = sgd(x,y, eta = 0.01, max_iter = iteration, batch_size = _batch_size)

                _title = f'SGD, batch size {_batch_size}'
                print( '\n\t'+_title)
                print ("Ein: ", Error(x,y,w))
                print ("Eout: ", Error(x_test, y_test, w))
                evaluationMetrics (x,y,w, '\nEvaluating output training data set')
                evaluationMetrics (x_test, y_test, w, '\nEvaluating output test data set')
                #plotResults(x,y,w, title = _title)
                multiple_w.append(w)
                multiple_title.append(_title)
        
                STOP_EXECUTION_TO_SEE_RESULT()

        plotResultMultiplesLines(x,y,multiple_w,f'Comparative batch sizes, {iteration} iterations',multiple_title)        

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


def noisyVector(y, percent_noisy_data):
        '''
        y target vector to introduce noise
        size_training_example: number of point generated in each experiment,
        percent_noisy_data
        '''
        len_y = len(y)
        
        index = list(range(len_y))
        np.random.shuffle(index)
        
        size_noisy_data = int((len_y*percent_noisy_data)/ 100 )

        noisy_y = np.copy(y)
        for i in index[:size_noisy_data]:
                noisy_y[i] *= -1
        return noisy_y

#labels 
y = np.array( [f(x[0],x[1]) for x in training_sample ])


percent_noisy_data = 10.0

y = noisyVector(y,  percent_noisy_data)

    
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
	
	index = np.where(y == l)
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
batch_size = 5
maximum_number_iterations = 1000


x = np.array( [
        np.array([ 1, x_n[0], x_n[1] ])
        for x_n in training_sample
])

y = np.array( [f(x_n[0],x_n[1]) for x_n in training_sample ])
y = noisyVector(y,  percent_noisy_data)
w = sgd_exact_number_iter(x, y, eta, maximum_number_iterations, batch_size )


_title = f'SGD, batch size {batch_size}'
print( '\n\t'+_title)
print ("Ein: ", Error(x,y,w))
evaluationMetrics (x,y,w, '\nEvaluating output training data set')

STOP_EXECUTION_TO_SEE_RESULT()

plt.clf()

for l in labels:
	
	index = np.where(y == l)
	plt.scatter(training_sample[index, 0],
                    training_sample[index,1],
                    c=colors[l],
                    label=str(l))

plt.title('Linear regression fit')
plt.xlabel('$x_1$ value')
plt.ylabel('$x_2$ value')

# regression line
# x_0 = -1
y_0 = (w[1] - w[0]) /w[2]
#x_1 = 1
y_1 = -( w[1]+w[0]) /w[2]

plt.plot([-1, 1], [y_0, y_1], 'k-', label=('SGD regression'))
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()

STOP_EXECUTION_TO_SEE_RESULT()

## d
print('\n EXPERIMENT (d), lineal regression\n')

        
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
                test_y = noisyVector(test_y, percent_noisy_data)
        
                y = np.array( [f(x[0],x[1]) for x in training_sample ])
                y = noisyVector(y,  percent_noisy_data)
 
                # fit
                x = np.array( [
                        featureVector(x_n)
                        for x_n in training_sample
                ])

                x_test = np.array( [
                        featureVector(x_n)
                        for x_n in test_sample
                ])
                
                w = sgd_exact_number_iter(x, y, eta, maximum_number_iterations, batch_size = 32)

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
batch_size = 5
maximum_number_iterations = 1000

def quadraticFeatureVector(x_n):
        '''
        INPUT 
         xn = (x1,x2) vector of coordinates 
        
        '''
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
y = np.array( [f(x[0],x[1]) for x in training_sample ])
y = noisyVector(y,  percent_noisy_data)


for i in [10, 50, 100, 200, 500, 700, 1000]:
        maximum_number_iterations = i
        w = sgd_exact_number_iter(x, y, eta, maximum_number_iterations , batch_size = 17)

        print('\nFor one experiment:')
        _title = f'SGD, batch size {batch_size}, number iterations {maximum_number_iterations}'
        print( '\n',_title)
        print ("Ein: ", Error(x,y,w))
        evaluationMetrics (x,y,w, '\nEvaluating output training data set')
print('')

STOP_EXECUTION_TO_SEE_RESULT()


## plotting

def equation (x,y,w):
        '''
        INPUT 
        x coordinate
        y coodinate 
        w weights vector
        
        OUTPUT
        Real number, the scalar product of features vector dot weights vector
        '''
        return ( w[0,0]
                 + w[1,0] * x
                 + w[2,0] * y
                 + w[3,0] * x * y
                 + w[4,0] * x**2
                 + w[5,0] * y**2
                )
'''
PLOTING LINEAR REGRESSION 

We are going to plot the (x,y) \in [-1,-1]^2 that their value after 
the linear regression for classification  is near to 0. 

That means that they are in the limit area.  
'''
error = 10**(-2.1)
space = np.linspace(-1,1,100)

z = [[ equation(i,j,w) for i in space] for j in space ]

plt.contour(space,space, z, 0, colors=['black'],linewidths=2 )

for l in labels:
	
	index = np.where(y == l)
	plt.scatter(training_sample[index, 0],
                    training_sample[index,1],
                    c=colors[l],
                    label=str(l))

plt.title('Quadratic regression fit')
plt.xlabel('$x_1$ value')
plt.ylabel('$x_2$ value')
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




"""
Exercise 3 
Author: Blanca Cano Camarero
"""

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