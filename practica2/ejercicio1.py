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
        labels: son las etiquetas posibles que queremos que distinga para colorear, 
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
        plt.legend()
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
x = simula_gaus(N, dimension,sigma_gauss)
scatter_plot(x, plot_title = f'Nube de puntos Gausiana (N = {N}, sigma = {sigma_gauss}) ')


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

stop('Apartado 1.2.a')
# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

#1.2 a)

stop('Apartado 1.2.a')
# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1


def f(x, y, a, b):
	return signo(y - a*x - b)


def f_sin_signo (x,y,a,b):
        return y - a*x - b

# datos del problema
rango = [-50, 50]
N = 100
dimension = 2

x = simula_unif(N, dimension,rango)

a,b=simula_recta(rango)

print("Los coeficientes a y b: ", a, b)
y = [ f(v[0],v[1],a,b) for v in x ]




# datos representación
labels = [-1,1]
colors = {-1: 'royalblue', 1: 'limegreen'}



classified_scatter_plot(x,y,
                        lambda x,y: f_sin_signo(x,y,a,b),
                        f'Apartado 2.a sin ruido',
                        labels, colors)




stop('Apartado 1.2.b')

def analisis_clasificado(y_obtenida, y_target):
        '''
        Imprime y devuelve además la precisión 
        '''
        diferencias = (y_target - y_obtenida) / 2 # el dos proviene de que las etiquetas son +1 y -1
        positivos_fallados =  sum(filter (lambda x: x>0, diferencias))
        negativos_fallados =  abs(sum(filter (lambda x: x<0, diferencias)))

        numero_positivos = sum(filter (lambda x: x>0, y_target))
        numero_negativos = abs(sum(filter (lambda x: x<0, y_target)))
                               
        porcentaje_positivos_fallados = positivos_fallados /numero_positivos * 100
        porcentaje_negativos_fallados = negativos_fallados /numero_negativos * 100

        total_fallados =  positivos_fallados +  negativos_fallados
        numero_total =  numero_positivos +   numero_negativos
        porcentaje_fallado_total = total_fallados / numero_total * 100

        precision = 100 - porcentaje_fallado_total
        
        print('Resultado clasificación: ')
        
        print(f'\t Positivos fallados {positivos_fallados}',
              f' de {numero_positivos},',
              f'lo que hace un porcentaje de {porcentaje_positivos_fallados}'
              )
        
        print(f'\t Negativos fallados {negativos_fallados}',
              f' de {numero_negativos},',
              f'lo que hace un porcentaje de {porcentaje_negativos_fallados}'
              )

        print(f'\t Total fallados {total_fallados}',
              f' de {numero_total},',
              f'lo que hace un porcentaje de {porcentaje_fallado_total}'
              )
        print(f'\t La precisión es ',
              f' de {100 - porcentaje_fallado_total} %'
              )

        return precision

                               

def noisyVector(y, percent_noisy_data, labels):
        '''
        y vector sobre el que introducir ruido
q        labels: etiquetas sobre las que vamso a introducir ruido
        percent_noisy_data: porcentaje de ruido de cada etiqueta , debe ser menro o igual que 100
        
        '''
        
        noisy_y = np.copy(y)
        
        for l in labels:
                index = [i for i,v in enumerate(y) if v == l]
                np.random.shuffle(index)
                len_y = len(index)
                size_noisy_data = round((len_y*percent_noisy_data)/ 100 )

        
                for i in index[:size_noisy_data]:
                        noisy_y[i] *= -1
        return noisy_y

porcentaje_ruido = 10 # por ciento 
noisy_y = noisyVector(y, porcentaje_ruido, labels)
precision = analisis_clasificado(noisy_y, y)

classified_scatter_plot(x,noisy_y,
                        lambda x,y: f_sin_signo(x,y,a,b),
                        f'Apartado 2.b $f$ ruidosa, con precisión del {precision}%',
                        labels, colors)



stop('Apartado 2.c')

funciones = [
        lambda x,y: (x -10)** 2 + (y-20)** 2 - 400,
        lambda x,y: 0.5*(x +10)** 2 + (y-20)** 2 - 400,
        lambda x,y: 0.5*(x -10)** 2 - (y+20)** 2 - 400,
        lambda x,y: y - 20* x**2 - 5 * x + 3
]

funciones_en_latex = [
        '$f(x,y) = (x-10)^2 + (y -20) ^2 -400$',
        '$f(x,y) =  0.5(x +10) ^ 2 + (y-20)^ 2 - 400$',
        '$f(x,y) =  0.5(x -10)^ 2 - (y-20)^ 2 - 400$',
        '$f(x,y) =  y - 20 x^2 - 5 x + 3$'
        ]


for i in range(len(funciones)):
        print(f'\nPara {funciones_en_latex[i]}')
        y_ajustado = np.array([ signo( funciones[i](v[0], v[1]))   for v in x])
        precision = analisis_clasificado(y_ajustado, noisy_y)
        classified_scatter_plot(x,noisy_y,
                                funciones[i],
                                'Clasificación para '+funciones_en_latex[i] + f' Precisión del {precision}%',
                                labels, colors)

        


