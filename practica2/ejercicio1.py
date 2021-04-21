# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante:  Blanca Cano Camarero  
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b


##############################################
# Mis funciones auxiliares 
############################################
def stop(apartado_siguiente = None):
        '''
        Detiene la ejecución
        si apartado siguiente tiene nombre lo muestra en panalla
        '''
        #input("\n--- Pulsar tecla para continuar ---\n")
        print("\n--- Pulsar tecla para continuar ---\n")
        if(apartado_siguiente):
                print(apartado_siguiente)


def scatter_plot(x, plot_title):
        '''Representa un scatter plot 
        x es una muestra
        '''
        plt.clf()

        plt.scatter(x[:, 0], x[:, 1], c = 'b')
        plt.title(plot_title)

        #plt.show()
        #uncoment

def classified_scatter_plot(x,y, function, plot_title, labels, colors):
        '''Dibuja los datos x con sus respectivas etiquetas y
        Dibuja la función: function 
        (todo esto en el mismo gráfico  
        '''
        plt.clf()

        
        for l in labels:
                index = [i for i,v in enumerate(y) if v == l]
                plt.scatter(x[index, 0], x[index, 1], c = colors[l], label = str(l))
                
                

        ## ejes
        xmin, xmax = np.min(x[:, 0]), np.max(x[:, 0])
        ymin, ymax = np.min(x[:, 1]), np.max(x[:, 1])

        
        ## function plot
        spacex = np.linspace(xmin,xmax,100)
        spacey = np.linspace(ymin,ymax,100)
        z = [[ function(i,j) for i in spacex] for j in spacey ]
        plt.contour(spacex,spacey, z, 0, colors=['red'],linewidths=2 )

        # título 
        plt.title(plot_title)

        plt.show()
        
############################################


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

stop('Apartado 1.1.a')
N = 50
dimension  = 2
rango_uniforme = [-50, 50]
x = simula_unif(N, dimension, rango_uniforme)
scatter_plot(x, plot_title = f'Nube de puntos uniforme (N = {N}, rango = {rango_uniforme})')

stop('Apartado 1.1.b')
sigma_gauss =  [5,7]
x = simula_gaus(N, dimension,rango_gauss)
scatter_plot(x, plot_title = f'Nube de puntos Gausiana (N = {N}, sigma = {sigma_gauss}) ')


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente
stop('Apartado 1.2')
# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1


def f(x, y, a, b):
	return signo(y - a*x - b)


def f_a (x,y,a,b):
        return y - a*x - b

# datos del problema
rango = [-50, 50]
N = 100
a,b = simula_recta(rango)
dimension = 2

x = simula_unif(N, dimension, rango)

y = [ signo(f(v[0],v[1],a,b)) for v in x ]




# datos representación
labels = [-1,1]
colors = {-1: 'blue', 1: 'yellow'}

def function (x,y):
        return f_a(x,y,a,b)

classified_scatter_plot(x,y,function, f'Apartado 2.a f(x,y) = y - {a}x -{b}', labels, colors)
