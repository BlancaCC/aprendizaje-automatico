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

        plt.show()
        

def classified_scatter_plot(x,y, function, plot_title, labels, colors):
        '''
        Dibuja los datos x con sus respectivas etiquetas
        Dibuja la función: function 
        y: son las etiquetas posibles que se colorearán
        labels: Nombre con el que aparecerán las etiquetas
        colors: colores de las diferentes etiquetas

        Todo lo dibuja en un gráfico
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


def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    


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

def getPrecision(y_obtenida, y_target):
        '''
        Imprime y devuelve además la precisión 
        '''
        diferencias = (y_target - y_obtenida) / 2 # el dos proviene de que las etiquetas son +1 y -1

        fallados = sum( map(abs, diferencias))
        total = len(y_target)

        precision = (1 - fallados/total)*100
        
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

plot_datos_cuad(x,noisy_y,
                lambda x: np.array([ f_sin_signo(v[0],v[1],a,b) for v in x]),
                title=f'Apartado 2.b $f$ ruidosa, con precisión del {precision}%',
                xaxis='x axis', yaxis='y axis')



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

        # gráficas
        classified_scatter_plot(x,noisy_y,
                                funciones[i],
                                'Clasificación para '+funciones_en_latex[i] + f' Precisión del {precision}%',
                                labels, colors)

        plot_datos_cuad(x,y_ajustado,
                lambda x: np.array([signo( funciones[i](v[0], v[1])) for v in x]),
                title='Clasificación para '+funciones_en_latex[i] + f' Precisión del {precision}%',
                xaxis='x axis', yaxis='y axis')









###############################################################################
###############################################################################
###############################################################################
stop('EJERCICIO 2.1 ALGORITMO PERCEPTRON')

# EJERCICIO 2.1: ALGORITMO PERCEPTRON



def ajusta_PLA(datos, labels, max_iter, vector_inicial):
    '''
    DATOS DE ENTRADA
    datos: matriz donde cada item y etiqueta es una fila 
    labels: vector de eitquetas +- 1 
    max_iter: número máximo de iteraciones 
    valor iniciar del vector (vector fila)

    SALIDA
    w: ajustado 
    paso: pasos usados para alcanzar la solución devuelta
    '''

    w = np.copy(vector_inicial) 
    w = w.T
    optimo  = False
    paso = 0
    
    while not optimo  and paso < max_iter:
            optimo = True
            paso += 1

            for i,x in enumerate(datos):
                    if signo(x.dot(w)) != labels[i]:
                            optimo = False
                            w = w + labels[i]*x.T #transpones 
             
    return w , paso


## Funciones para el formateo rápido
to_round =  lambda x : float(str('%.3f' % round(x,3)))
                                  
def to_round_list(l):
        '''
        Dada una lista de flotantes, la devuelve redondeada a tres decimales
        '''
        return list( map(to_round, l))

H = lambda w: (lambda x,y: w[0] + x*w[1] + y*w[2])

print('Ejercicio 2.a.1')


# datos del problema
rango = [-50, 50]
N = 100
dimension = 2



x_redimensionada = np.c_[np.ones(len(x)),x]

print('Apartado 2.a.1 a) vector inicial nulo')

w_inicial = np.array([0.0, 0.0, 0.0])

w_final, pasos = ajusta_PLA(x_redimensionada, y,
                            max_iter=100, vector_inicial= w_inicial)

print(f'Tras ajustar el vector final es { w_final} tras {pasos} pasos' )


# Comparativas con recta inicial
print(f'Los coeficientes recta originaria y = ax +b son: a = {a}, b = {b}')
print(f'Mientras que los nuestros son a ={-w_final[1]/w_final[2]}, b = {-w_final[0]/w_final[2]}')

# veamso si ajusta bien
h = H( w_final)

classified_scatter_plot(x,y,
                        h,
                        '2.a.1 Ajuste perceptrón, vector nulo',
                        labels,
                        colors)

plot_datos_cuad(x,y,
                lambda x: np.array([signo( h(v[0], v[1])) for v in x]),
                title='2.a.1 Ajuste perceptrón, vector nulo',
                xaxis='x axis', yaxis='y axis')


veces_experimento = 10
stop(f'Apartado 2.a.1 b) Con vectores aleatorios en [0,1] {veces_experimento}')

sucesion_pasos = []
sucesion_wf = []
sucesion_w0 = []
print('numero_pasos | \t w_0 \t |\t w_f')
for i in range(veces_experimento):
        w_0 = simula_unif(3, 1, [0, 1]).reshape(1, -1)[0]

        w_f, numero_pasos = ajusta_PLA(x_redimensionada, y,
                                       max_iter=500, vector_inicial= w_0)
        sucesion_pasos.append(numero_pasos)
        sucesion_wf.append(w_f)
        sucesion_w0.append(w_0)
        print(numero_pasos, ' | ',
              to_round_list(w_0), ' |',
              to_round_list(w_f))

        


print('El número de pasos necesario en cada iteración es de:\n',  sucesion_pasos)
#print('El vector inicial para cada iteración, ha sido: \n', sucesion_w0)
#print('El vector final para cada iteración, ha sido: \n', sucesion_wf)

# vamos a pintar la solución del primero y la del último

h = lambda w : (lambda x,y : w[0] + x*w[1] + y*w[2])
# cogemso estos índices de marnera arbitraria porque son 'diferentes'
#for i in [0,-1]:
        

media = np.mean(sucesion_pasos)
desviacion_tipica = np.std(sucesion_pasos)

print(f'Tiene una media de {media} y desviación típica {desviacion_tipica}')


stop(f'Apartado 2.a.2 ) Repetición experimento con valores de 2b')

iteraciones_maximas = [100, 200, 300]
for j in iteraciones_maximas:
        print(f'\nPara max_iter = {j}:  ')
        sucesion_pasos = []
        sucesion_wf = []
 
        print('numero_pasos | \t w_0 \t |\t w_f | \t Precisión (%)  ')
        print(' --- |  ---  |--- |  ---  ')
        for i in range(veces_experimento):
                w_0 = sucesion_w0[i]

                w_f, numero_pasos = ajusta_PLA(x_redimensionada, noisy_y,
                                               max_iter= j,
                                               vector_inicial= w_0)
                y_obtenida = [signo((H(w_f))(v[0],v[1])) for v in x]
                sucesion_pasos.append(numero_pasos)
                sucesion_wf.append(w_f)
                
                print(numero_pasos, ' | ',
                      to_round_list(w_0), ' |',
                      to_round_list(w_f), '|', 
                      to_round(getPrecision(y_obtenida, noisy_y)),
                      '  '
                      )




stop('\nEjercicio 1.2.b _____  Regresión logística ______\n')



def funcionLogistica(x, w):
        s = w.dot(x.T)
        return np.exp(s) / (1 + np.exp(s))


def gradienteLogistica(x,y,w):
        '''
        '''
        '''
        n = len(y)
        return  -1/n*sum([ y[i]* x[i] / (1 + np.exp( y[i]* x[i].dot( w.T))    for i in range(n)])

        '''
        return -(y * x)/(1 + np.exp(y * w.dot(x.T)))


def errorRegresionLogistica(x,y,w):

        n = len(y)
        sumatoria = 0
        productorio = y*x.dot(w.T)
        for i in productorio:
                sumatoria += np.log(1 + np.exp(- i))
        
        return  sumatoria / n

        
                         
def regresionLogistica(x, y, tasa_aprendizaje, w_inicial, max_iteraciones, tolerancia):
        '''
        Regresión logística(LR) con Gradiente Descendente Estocástico (SGD)
        '''

        w_final = np.copy(w_inicial)
        w_nueva = np.copy(w_inicial)
        
        iteraciones = 0
        tolerancia_actual  = np.inf
        
        n = len(y)
        indices = np.arange(n)
        
        while (iteraciones < max_iteraciones and
               tolerancia_actual > tolerancia):

                indices = np.random.permutation(indices)
                for i in indices:
                        # tamaño del bath uno
                        w_nueva = w_nueva - tasa_aprendizaje * gradienteLogistica(x[i],y[i],w_nueva)

                tolerancia_actual = np.linalg.norm(w_nueva - w_final) 
                w_final = np.copy(w_nueva)

                iteraciones += 1
                
        return  w_final , iteraciones


### 2b Experimento


tamano_muestra_2b = 100
intervalo_2b = [0.0, 2.0]
x_2b =  simula_unif( tamano_muestra_2b, 2, intervalo_2b )
x_2b_redimensionado = np.c_[np.ones(tamano_muestra_2b), x_2b]
#np.array([ np.array([1,v[0],v[1]]) for v in x_2b])

puntos_recta = np.random.choice(tamano_muestra_2b, 2, replace = False)


#coordenadas
x_1 = x_2b [puntos_recta[0]][0]
y_1 = x_2b [puntos_recta[0]][1]
x_2 = x_2b [puntos_recta[1]][0]
y_2 = x_2b [puntos_recta[1]][1]

# cálculos recta

pendiente =  (y_2 - y_1) / (x_2 - x_1)
ordenada_origen = y_2 - pendiente * x_2

# ecuación punto punto pendiente # y = mx + a

recta = lambda x,y :  y - (x*pendiente + ordenada_origen)

y_2b = [np.sign( recta(v[0], v[1]))  for v in x_2b]

classified_scatter_plot(x_2b,
                        y_2b,
                        recta,
                        'Nube puntos experimento 2b',
                        labels, colors)

stop ('Vamos a determinar la w para este caso')


w_2b, epocas = regresionLogistica(x_2b_redimensionado,
                                  y_2b,
                                  tasa_aprendizaje = 0.01,
                                  w_inicial =np.array([0.0 for i in range(3)]),
                                  max_iteraciones = np.inf, # por ser separable convergirá
                                  tolerancia = 0.01)


print(f'El vecto al que converge es {w_2b}, tras {epocas} epocas')
E_in = errorRegresionLogistica(x_2b_redimensionado,y_2b,w_2b)

print(f'E_in = {E_in}')

n_test  = 1000
x_test = simula_unif( n_test, 2, intervalo_2b )
y_test = np.array([np.sign(recta(v[0], v[1])) for v in x_test])

x_test_redimensionado =  np.c_[np.ones(n_test), x_test]
#np.array([ np.array([1,v[0],v[1]]) for v in x_test])

E_out = errorRegresionLogistica(x_test_redimensionado,y_test,w_2b)

print(f'E_out = {E_out}')
h = lambda x,y :np.sign(np.array([1,x,y]).dot(w_2b))
y_test_obtenida = [h(v[0], v[1]) for v in x_test]

analisis_clasificado(y_test_obtenida, y_test)


classified_scatter_plot(x_test,y_test,
                        lambda x,y :np.array([1,x,y]).dot(w_2b),
                        f'Ajuste test regresión logística test, tras {epocas} épocas, $Eout$ = {to_round(E_out)} ',
                        labels, colors)
                          


#aunque sean separables, nunca sabremos si ajusta  a la buena
# cota hoefding
repeticiones_experimento = 100
stop(f'Repetimos el experimentos {repeticiones_experimento} veces')

precisiones = np.empty(repeticiones_experimento)
pasos = np.empty(repeticiones_experimento)
errores = np.empty(repeticiones_experimento)

h_experimento =  lambda x,y, w:np.sign(np.array([1,x,y]).dot(w))
w_0 = np.zeros(3)

for i in range(repeticiones_experimento):

        # generamso los datos
        # entrenamiento
        x =  simula_unif( tamano_muestra_2b, 2, intervalo_2b )
        x = np.c_[np.ones(tamano_muestra_2b), x]
        y = np.array([np.sign( recta(v[1], v[2]))  for v in x])

        # test
        x_test = simula_unif( n_test, 2, intervalo_2b )
        y_test = np.array([np.sign(recta(v[0], v[1])) for v in x_test])


        # entrenamos
        w, pasos[i] = regresionLogistica(x,
                                  y,
                                  tasa_aprendizaje = 0.01,
                                  w_inicial = w_0,
                                  max_iteraciones = np.inf, # por ser separable convergirá
                                  tolerancia = 0.01)

        
        y_obtenida = [h_experimento(v[0], v[1], w) for v in x_test]
        precisiones[i] = getPrecision(y_obtenida, y_test)
        errores[i] = errorRegresionLogistica(np.c_[np.ones(n_test), x_test],
                                             y_test,
                                             w)
        if i % 10 == 0:
                print(f'{i+1} experimentos de {repeticiones_experimento}')
        

# medias 
media_epocas = np.mean(pasos)
desviacion_tipica_epocas = np.std(pasos)

media_errores = np.mean(errores)
desviacion_tipica_errores = np.std(errores)

media_precisiones = np.mean(precisiones)
desviacion_tipica_precisiones = np.std(precisiones)

print('\nResultados sin redondear')
print(f'El número medio de épocas es { media_epocas}, con desviación típica { desviacion_tipica_epocas}')
print(f'El E_out medio es { media_errores}, con desviación típica { desviacion_tipica_errores}')
print(f'La precisión media es {media_precisiones}, con desviación típica {desviacion_tipica_precisiones}')

print('\nResultados redondeados a tres decimales')
print(f'El número medio de épocas es {to_round(media_epocas)}, con desviación típica {to_round(desviacion_tipica_epocas)}')
print(f'El E_out medio es {to_round(media_errores)}, con desviación típica {to_round(desviacion_tipica_errores)}')

print(f'La precisión media es {to_round(media_precisiones)}, con desviación típica {to_round(desviacion_tipica_precisiones)}')



        

stop('BONUS')


####################################################################################################
# BONUS

####################################################################################################
label8 = 1
label4 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
        # Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 8 la 4
	for i in range(0,datay.size):
		if datay[i] == 8 or datay[i] == 4:
			if datay[i] == 8:
				y.append(label8)
			else:
				y.append(label4)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y



# REGRESIÓN LINEAL
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




## Gráfica para comparar dos líneas


def comparaValoresGrafica (x, y_1, y_2, etiqueta_1, etiqueta_2, etiqueta_x, etiqueta_y, titulo):

        plt.clf()
        plt.title(titulo)
        
        plt.plot(x, y_1, label=etiqueta_1,
                 linestyle = 'solid', color = 'blue')
        plt.plot(x, y_2, label=etiqueta_2,
                  linestyle = 'solid', color = 'mediumorchid')
        plt.xlabel(etiqueta_x)
        plt.ylabel(etiqueta_y)
        plt.legend()
        plt.show()


def muestraTabla (titulos, columnas):

        print(' | '.join(titulos) + '\t  ')

        n = len(titulos)

        print( (n-1)*((3*'-') + '|') + 3*'-')

        columnas = np.array(columnas)
        
        for f in columnas.T:
                print(' | '.join((map(str, map(to_round,f)))) + '\t  ')
                
        
        

        


# Reading training data set 
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Reading test data set 
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')




batch_sizes =[32] #batch sizes compared in the experiment

n_iterations = [10,20,50,100,200,500,100]
titulos = ['Iteraciones', 'Ein', 'Eout', 'Precision In', 'Precision out']
len_iter = len(n_iterations)

Ein_SGD = np.empty(len_iter)
Eout_SGD = np.empty(len_iter)
accuracy_in_SGD = np.empty(len_iter)
accuracy_out_SGD = np.empty(len_iter)

w_SGD = []

for _batch_size in batch_sizes:
        print('Para SGD de tamañ de batch {_batch_size}')

        for i, iteration in enumerate(n_iterations):
                w_SGD = sgd(x,y, eta = 0.01, max_iter = iteration, batch_size = _batch_size)
                Ein_SGD[i] =  Error(x,y,w_SGD)
                Eout_SGD[i] =   Error(x_test, y_test, w_SGD)

                y_obtenida_entrenamiento =np.sign(x.dot(w_SGD).T)[0]
                accuracy_in_SGD[i] = getPrecision(y_obtenida_entrenamiento, y)
                
                y_obtenida = np.sign(x_test.dot(w_SGD).T)[0]
                accuracy_out_SGD[i] = getPrecision(y_obtenida, y_test)


        ### Muestra de los datos
        
        #comparación Ein_SGD Eout_SGD
        comparaValoresGrafica (x = n_iterations,
                               y_1 = Ein_SGD, y_2 = Eout_SGD,
                               etiqueta_1 = '$E_{in}$', etiqueta_2 ='$E_{out}$',
                               etiqueta_x = 'Iteraciones',
                               etiqueta_y = 'Error',
                               titulo = f'Errors para SGD, tamaño de batch {_batch_size}'
                               )

        comparaValoresGrafica (x = n_iterations,
                               y_1 = accuracy_in_SGD, y_2 = accuracy_out_SGD,
                               etiqueta_1 = 'Precisión entrenamiento',
                               etiqueta_2 ='Precisión test',
                               etiqueta_x = 'Iteraciones',
                               etiqueta_y = 'Precisión',
                               titulo = f'Precisión para SGD, tamaño de batch {_batch_size}'
                               )
        muestraTabla (titulos, [n_iterations, Ein_SGD, Eout_SGD, accuracy_in_SGD, accuracy_out_SGD])

        print('Grafica para clasificación test SGD, con zona de clasificación')
        plot_datos_cuad(x_test[:, 1:],y_test,
                #lambda x: np.sign( np.c_[np.ones(len(x), x)].dot(w_SGD)),
                lambda x: np.array([signo( np.array([1,v[0], v[1]]).dot(w_SGD)) for v in x]),
                title=f'Clasificación para SGD {n_iterations[-1]} iteraciones, Eout = {to_round(Eout_SGD[-1])}, precisión {to_round(accuracy_out_SGD[-1])}',
                xaxis='Intensidad media', yaxis='Simetría media')

        print('\nGrafica para clasificación test SGD, sin zona de clasificación')

        classified_scatter_plot(x_test[:, 1:],y_test,
                                lambda x,y : np.array([1,x,y]).dot(w_SGD)[0],
                                f'Clasificación para SGD {n_iterations[-1]} iteraciones, Eout = {to_round(Eout_SGD[-1])}, precisión {to_round(accuracy_out_SGD[-1])}',
                                labels, colors)

#-------------------------------------------------------
stop('Clasificación para PLA')
# Utilizaremos ahora el algoritmo PLA

Ein_PLA = np.empty(len_iter)
Eout_PLA = np.empty(len_iter)
accuracy_in_PLA = np.empty(len_iter)
accuracy_out_PLA = np.empty(len_iter)
w_PLA = []
for i, iteration in enumerate(n_iterations):
        
        w_PLA , _ = ajusta_PLA(x, y,
                   max_iter=iteration, vector_inicial= np.zeros(3))
        Ein_PLA[i] =  Error(x,y,w_PLA)
        Eout_PLA[i] =   Error(x_test, y_test, w_PLA)
        
        y_obtenida_entrenamiento = np.sign(x.dot(w_PLA))
        accuracy_in_PLA[i] = getPrecision(y_obtenida_entrenamiento, y)
                
        y_obtenida = np.sign(x_test.dot(w_PLA))
        accuracy_out_PLA[i] = getPrecision(y_obtenida, y_test)


        ### Muestra de los datos
        
        #comparación Ein_PLA Eout_PLA
comparaValoresGrafica (x = n_iterations,
                       y_1 = Ein_PLA, y_2 = Eout_PLA,
                       etiqueta_1 = '$E_{in}$', etiqueta_2 ='$E_{out}$',
                       etiqueta_x = 'Iteraciones',
                       etiqueta_y = 'Error',
                       titulo = f'Errors para PLA'
                       )

comparaValoresGrafica (x = n_iterations,
                       y_1 = accuracy_in_PLA, y_2 = accuracy_out_PLA,
                       etiqueta_1 = 'Precisión entrenamiento',
                       etiqueta_2 ='Precisión test',
                       etiqueta_x = 'Iteraciones',
                       etiqueta_y = 'Precisión',
                       titulo = 'Precisión para PLA'
                               )
muestraTabla (titulos, [n_iterations, Ein_PLA, Eout_PLA, accuracy_in_PLA, accuracy_out_PLA])

## Dibujamos gráficas
print('Grafica para clasificación test PLA, con zona de clasificación')
plot_datos_cuad(x_test[:, 1:],y_test,
                lambda x: np.array([signo( np.array([1,v[0], v[1]]).dot(w_PLA.T)) for v in x]),
                title=f'Clasificación para PLA {n_iterations[-1]} iteraciones, Eout = {to_round(Eout_PLA[-1])}, precisión {to_round(accuracy_out_PLA[-1])}',
                xaxis='Intensidad media', yaxis='Simetría media')

print('\nGrafica para clasificación test PLA, sin zona de clasificación')

classified_scatter_plot(x_test[:, 1:],y_test,
                        lambda x,y : np.array([1,x,y]).dot(w_PLA.T),
                        f'Clasificación para PLA {n_iterations[-1]} iteraciones, Eout = {to_round(Eout_PLA[-1])}, precisión {to_round(accuracy_out_PLA[-1])}',
                        labels, colors)



stop('Comparamos errores del test')
# COMPARAMOS VALORES
comparaValoresGrafica (x = n_iterations,
                       y_1 = Eout_SGD, y_2 = Eout_PLA,
                       etiqueta_1 = '$E_{out} SGD$', etiqueta_2 ='$E_{out} PLA$',
                       etiqueta_x = 'Iteraciones',
                       etiqueta_y = '$E_{out}$',
                       titulo = 'Comparativa error $E_{out}$ PLA y SGD'
                       )



### Implementación de PLA-Pocket


stop('Clasificación para PLA-pocket')

def ajusta_PLA_pocket(x, y,  max_iter, vector_inicial):
    '''
    DATOS DE ENTRADA
    x: matriz donde cada item y etiqueta es una fila 
    y: vector de eitquetas +- 1 
    max_iter: número máximo de iteraciones 
    valor iniciar del vector (vector fila)

    SALIDA
    w: ajustado 
    paso: pasos usados para alcanzar la solución devuelta
    '''

    w = np.copy(vector_inicial) 
    w = w.T
    optimo  = False
    paso = 0

    w_mejor = np.copy(vector_inicial)
    y_obtenida = np.sign(x.dot(w_mejor))
    precision_mejor =getPrecision(y_obtenida, y)
    traza_precision = []
    
    while not optimo  and paso < max_iter:
        optimo = True
        paso += 1

        for i,v in enumerate(x):
                if signo(v.dot(w)) != y[i]:
                        optimo = False
                        w = w + y[i]*v.T #transpones

        # actualizamso si es mejor (tiene más precisión)
        y_obtenida = np.sign(x.dot(w))
        precision_nueva = getPrecision(y_obtenida, y)
        
        if(precision_nueva > precision_mejor):
                precision_mejor = precision_nueva
                w_mejor = w
        
        #traza_precision.append(precision_mejor)
          
        
    return w_mejor , paso, precision_mejor#traza_precision




# Utilizaremos ahora el algoritmo PLA_POCKET

Ein_PLA_POCKET = np.empty(len_iter)
Eout_PLA_POCKET = np.empty(len_iter)
accuracy_in_PLA_POCKET = np.empty(len_iter)
accuracy_out_PLA_POCKET = np.empty(len_iter)
w_PLA_POCKET = []
for i, iteration in enumerate(n_iterations):
        
        w_PLA_POCKET , _, accuracy_in_PLA_POCKET[i] = ajusta_PLA_pocket(x, y,
                   max_iter=iteration, vector_inicial= np.zeros(3))
        Ein_PLA_POCKET[i] =  Error(x,y,w_PLA_POCKET)
        Eout_PLA_POCKET[i] =   Error(x_test, y_test, w_PLA_POCKET)
                
        y_obtenida = np.sign(x_test.dot(w_PLA_POCKET))
        accuracy_out_PLA_POCKET[i] = getPrecision(y_obtenida, y_test)


        ### Muestra de los datos
        
        #comparación Ein_PLA_POCKET Eout_PLA_POCKET
comparaValoresGrafica (x = n_iterations,
                       y_1 = Ein_PLA_POCKET, y_2 = Eout_PLA_POCKET,
                       etiqueta_1 = '$E_{in}$', etiqueta_2 ='$E_{out}$',
                       etiqueta_x = 'Iteraciones',
                       etiqueta_y = 'Error',
                       titulo = f'Errors para PLA_POCKET'
                       )

comparaValoresGrafica (x = n_iterations,
                       y_1 = accuracy_in_PLA_POCKET, y_2 = accuracy_out_PLA_POCKET,
                       etiqueta_1 = 'Precisión entrenamiento',
                       etiqueta_2 ='Precisión test',
                       etiqueta_x = 'Iteraciones',
                       etiqueta_y = 'Precisión',
                       titulo = 'Precisión para PLA_POCKET'
                               )
muestraTabla (titulos, [n_iterations, Ein_PLA_POCKET, Eout_PLA_POCKET, accuracy_in_PLA_POCKET, accuracy_out_PLA_POCKET])

## Dibujamos gráficas
print('Grafica para clasificación test PLA_POCKET, con zona de clasificación')
plot_datos_cuad(x_test[:, 1:],y_test,
                lambda x: np.array([signo( np.array([1,v[0], v[1]]).dot(w_PLA_POCKET.T)) for v in x]),
                title=f'Clasificación para PLA_POCKET {n_iterations[-1]} iteraciones, Eout = {to_round(Eout_PLA_POCKET[-1])}, precisión {to_round(accuracy_out_PLA_POCKET[-1])}',
                xaxis='Intensidad media', yaxis='Simetría media')

print('\nGráfica para clasificación test PLA_POCKET, sin zona de clasificación')

classified_scatter_plot(x_test[:, 1:],y_test,
                        lambda x,y : np.array([1,x,y]).dot(w_PLA_POCKET.T),
                        f'Clasificación para PLA_POCKET {n_iterations[-1]} iteraciones, Eout = {to_round(Eout_PLA_POCKET[-1])}, precisión {to_round(accuracy_out_PLA_POCKET[-1])}',
                        labels, colors)
