import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print ('Exercise 2\n')




#### EXPERIMIENTS ###########

## a)

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

plt.clf()
plt.scatter(training_sample[:, 0], training_sample[:, 1], c='b')
plt.title('Muestra de entrenamiento generada por una distribución uniforme')
plt.title('Training sample generated by a uniform distribution')
plt.xlabel('$x_1$ value')
plt.ylabel('$x_2$ value')

plt.show()


## b)

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign(
            (x1 -0.2)**2
            +
            x2**2 -0.6
        ) 


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



#### C
