'''
PRÁCTICA 3 Refresión 
Blanca Cano Camarero   
'''
#############################
#######  BIBLIOTECAS  #######
#############################
# Biblioteca lectura de datos
# ==========================
import pandas as pd

# matemáticas
# ==========================
import numpy as np

# Modelos lineales de clasificación a usar   
# =========================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor


# Preprocesado 
# ==========================
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# visualización de datos
# ==========================
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns # utilizado para pintar la matriz de correlación 

# Validación cruzada
# ==========================
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

# metricas
# ==========================
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# Otros
# ==========================
from operator import itemgetter #ordenar lista
import time

np.random.seed(1)

########## CONSTANTES #########
NOMBRE_FICHEROS_REGRESION  = ['./datos/train.csv','./datos/unique_m.csv']
SEPARADOR_REGRESION = ','

NUMERO_CPUS_PARALELO = 4
####################################################
################### funciones auxiliares 
def LeerDatos (nombre_fichero, separador, cabecera = None):
    '''
    Input: 
    - file_name: nombre del fichero path relativo a dónde se ejecute o absoluto
    La estructura de los datos debe ser: 
       - Cada fila un vector de características con su etiqueta en la última columna.

    Outputs: x,y
    x: matriz de filas de vector de características
    y: vector fila de la etiquetas 
    
    '''

    datos = pd.read_csv(nombre_fichero,
                       sep = separador,
                        header = cabecera,
                        #low_memory = False # Para que no haya tipos de datos mezclados 
                        )
    valores = datos.values

    # Los datos son todas las filas de todas las columnas salvo la última 
    x = valores [:: -1]
    y = valores [:, -1] # el vector de características es la últma columana

    return x,y


def VisualizarClasificacion2D(x,y, titulo=None):
    """Representa conjunto de puntos 2D clasificados.
    Argumentos posicionales:
    - x: Coordenadas 2D de los puntos
    - y: Etiquetas"""

    _, ax = plt.subplots()
    
    # Establece límites
    xmin, xmax = np.min(x[:, 0]), np.max(x[:, 0])
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1)

    # Pinta puntos
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", alpha=0.8)

    # Pinta etiquetas
    etiquetas = np.unique(y)
    for etiqueta in etiquetas:
        centroid = np.mean(x[y == etiqueta], axis=0)
        ax.annotate(int(etiqueta),
                    centroid,
                    size=14,
                    weight="bold",
                    color="white",
                    backgroundcolor="black")

    # Muestra título
    if titulo is not None:
        plt.title(titulo)
    plt.show()


def Separador(mensaje = None):
    '''
    Hace parada del código y muestra un menaje en tal caso 
    '''
    print('\n-------- fin apartado, enter para continuar -------\n')

    if mensaje:
        print('\n' + mensaje)





#######################################################################
#######################################################################
#######################################################################


# Lectura de los datos

print(f'Vamos a proceder a leer los datos de los ficheros {NOMBRE_FICHEROS_REGRESION}')


x,y = LeerDatos( NOMBRE_FICHEROS_REGRESION[0], SEPARADOR_REGRESION, 0)

'''
for nombre_fichero in NOMBRE_FICHEROS_REGRESION[2:]:
    x_aux,y_aux = LeerDatos( nombre_fichero, SEPARADOR_REGRESION)
    np.append(x,x_aux)
    np.append(y,y_aux)

'''
Separador('Separamso test y entrenamiento')

###### separación test y entrenamiento  #####
ratio_test_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size= ratio_test_size,
    shuffle = True, 
    random_state=1)


print('Veamos ahora si los datos están bie distribuidos')
### Probar a ajustar con datos balanceados aunque menos


#vemos que están balanceados

def BalanceadoRegresion(y, divisiones = 20):
    min_y = min(y)
    max_y = max(y)

    longitud = (max_y - min_y)/divisiones    
    extremo_inferior = min_y
    extremo_superior = min_y + longitud

    datos_en_rango = np.arange(divisiones)
    cantidad_minima = np.infty
    cantidad_maxima = - np.infty
    indice_minimo = None
    indice_maximo = None
    
    
    for i in range(divisiones):
        datos_en_rango[i] = np.count_nonzero(
            (extremo_inferior <= y ) &
            (y <= extremo_superior)
        )
        extremo_inferior = extremo_superior
        extremo_superior += longitud

        if cantidad_minima > datos_en_rango[i]:
            cantidad_minima = datos_en_rango[i]
            indice_minimo = i
        if cantidad_maxima < datos_en_rango[i]:
            cantidad_maxima = datos_en_rango[i]
            indice_maximo = i

    # imprimimos valores
    #getcontext().prec = 4
    print('COMPROBACIÓN BALANCEO DE Y')
    
    print('Número total de etiquetas ', len(y))
    print('Rango de valores de y [%.4f, %.4f]'%(min_y, max_y))
    print('Cantidad mínima de datos ', cantidad_minima)
    extremo_inferior = min_y + longitud * indice_minimo
    print(f'Alcanzada en intervalo [%.4f , %.4f]'%
          (extremo_inferior , (extremo_inferior + longitud)))
    
    print('Cantidad máxima de datos ', cantidad_maxima)
    extremo_inferior = min_y + longitud * indice_maximo
    print(f'Alcanzada en intervalo [%.4f , %.4f]'%
          (extremo_inferior , (extremo_inferior + longitud)))
    print('La media es %.4f'% datos_en_rango.mean())
    print('La desviación típica %.4f' % datos_en_rango.std())

    # gráfico  de valores
    plt.title('Número de etiquetas por rango de valores')
    plt.bar([i*longitud + min_y for i in range(len(datos_en_rango))],datos_en_rango, width = longitud * 0.9)
    plt.xlabel('Valor de la etiqueta y (rango de longitud %.3f)'%longitud)
    plt.ylabel('Número de etiquetas')
    plt.show()
    

    
        
### Comprobación de balanceo 
Separador('Comprobamos balanceo')
BalanceadoRegresion(y, divisiones = 30)

restricciones_y = [100, 140]
for restriccion_y in restricciones_y: 
    Separador(f'Veamos para datos que cumplan y>{restriccion_y}')
    BalanceadoRegresion(y[y>restriccion_y], 30)


## Quitamos outliers




Separador('Normalizamos los datos')
#Normalizamos los datos para que tengan media 0 y varianza 1
scaler = StandardScaler()
#x_train_normalizados = scaler.fit_transform( x_train )
#x_test_normalizados = scaler.transform( x_test)

x_train = scaler.fit_transform( x_train )
x_test = scaler.transform( x_test) 


#------- correlacion ----
def PlotMatrizCorrelacion(matriz_correlacion):
    '''
    Muestra en pantalla la matriz de correlación de x
    usa la biblioteca de seaborn 
    '''
    plt.figure(figsize=(12,8))
    plt.title('Matriz de correlación')
    sns.heatmap(matriz_correlacion)
    plt.show()


    
def Pearson( x, umbral, traza = False):
    '''INPUT 
    x vector de caracteríscas 
    umbral: valor mínimo del coefiente para ser tenido en cuenta
    traza: Imprime coeficiente de Pearson e índices que guardan esa relación.   muestra gráfico: determian si se muestra una tabla con los coeficinetes que superen el umbral

    OUTPUT
    indice_explicativo: índice de columnas linealente independientes (con ese coeficiente)
    relaciones: lista de tuplas (correlacion, índice 1, índice 2)

    '''
    r = np.corrcoef(x.T)
    longitud_r  = len(r[0])
    # Restamos la matriz identidad con la diagonal
    # Ya que queremos encontrar donde alcanza buenos niveles de correlación no triviales 
    sin_diagonal = r - np.identity(len(r[0])) 
    relaciones = [] # guardaremos tupla y 


    # Devolveré un vector con lo índices que no puedan ser explicado,
    # Esto es, si existe una correlación mayor que el umbra entre i,j
    # y I es el conjunto de características el nuevo I = I -{j}
    # Denotarelos a I con la variable índice explicativo 
    indice_explicativo = np.arange( len(x[0]))


    # muestra tupla con el coefiente de pearson y los dos índices con ese vector de características
    for i in range(longitud_r):
        for j in range(i+1, longitud_r):
            if abs(sin_diagonal[i,j]) > umbral:
            
                relaciones.append((sin_diagonal[i,j], i,j))
                #print(sin_diagonal[i,j], i,j)

                indice_explicativo [j] = 0 # Indicamos que la columna j ya no es explicativa
                #para ello la pongo a cero, ya que el 0 siempre explicará, por ir  de menor a mayor los subíndices

    indice_explicativo = np.unique(indice_explicativo) # dejamos solo un cero

    
    relaciones.sort(reverse=True, key =itemgetter(0))

    # imprimimos las relaciones en orden
    if(traza):
        print(f'\nCoeficiente pearson para umbral {umbral}')
        print('Coeficiente | Índice 1 | Índice 2    ')
        print( '--- | --- | ---    ')
        for i,j,k in relaciones:
            print(i,' | ' , j, ' | ', k , '    ')

    return indice_explicativo, relaciones


Separador('Matriz de correlación asociada a los datos de entrenamiento')
PlotMatrizCorrelacion(np.corrcoef(x_train.T))


Separador('Índice de las características a mantener')

### Cálculos para distinto umbrales
umbrales = [0.999, 0.99, 0.98, 0.97, 0.95, 0.9]
indice_explicativo = dict()
relaciones = dict()

for umbral in umbrales:
    indice_explicativo[umbral], relaciones[umbral] = Pearson( x_train,
                                                              umbral,
                                                              traza = True,
                                                            )
numero_caracteristicas = len(x_train[0])
print(f'\nEl número inical de características es de { numero_caracteristicas}\n' )
print('Las reducciones de dimensión total son: \n')
print('| umbral | tamaño tras reducción | reducción total |    ')
print('|:------:|:---------------------:|:---------------:|    ')
for  umbral, ie in indice_explicativo.items():
    len_ie = len(ie)
    print(f'| {umbral} | {len_ie} | {numero_caracteristicas - len_ie} |    ')



    
umbral_seleccionado = 0.97 # debe de estar definidio en la lista umbrales  

def ReducirCaracteristicas(x,indices_representativos):
    '''
    x vector características
    indices_representativos: índices características que mantener

    OUTPUT 
    x_reducido 
    '''
    x_reducido = (x.T[indices_representativos]).T

    return x_reducido


## Nos queamos con los datos que 
x_train_reducido = ReducirCaracteristicas(x_train, indice_explicativo[ umbral_seleccionado])
x_test_reducido = ReducirCaracteristicas(x_test, indice_explicativo[ umbral_seleccionado])


## ¿aplicamso PCA ? 


### Validación cruzada
def MostrarMatrizConfusion(clasificador, x, y, titulo, normalizar):
    '''
    normalizar: 'true' o 'false', deben de ser los valores de normalice en mostrar_plot
    '''
    
    mostrar_plot = plot_confusion_matrix(clasificador,
                                         x , y,
                                         normalize = normalizar)
    mostrar_plot.ax_.set_title(titulo)
    plt.show()


def Evaluacion( clasificador,
                x, y, x_test, y_test,
                k_folds,
                nombre_modelo,
                metrica_error):
    '''
    Función para automatizar el proceso de experimento: 
    1. Ajustar modelo.
    2. Aplicar validación cruzada.
    3. Medir tiempo empleado en ajuste y validación cruzada.
    4. Medir la precisión.   

    INPUT:
    - Clasificador: Modelo con el que buscar el clasificador
    - X datos entrenamiento. 
    - Y etiquetas de los datos de entrenamiento
    - x_test, y_test
    - k-folds: número de particiones para la validación cruzada
    - metrica_error: debe estar en el formato sklearn (https://scikit-learn.org/stable/modules/model_evaluation.html)

    OUTPUT:
    '''

    ###### constantes a ajustar
    numero_trabajos_paralelos_en_validacion_cruzada = NUMERO_CPUS_PARALELO
    ##########################
    
    print('\n','-'*20)
    print (f' Evaluando {nombre_modelo}')
    #print('-'*20)

    
    #print(f'\n------ Ajustando modelo------\n')        
    tiempo_inicio_ajuste = time.time()
    
    #ajustamos modelo 
    ajuste = clasificador.fit(x,y) 
    tiempo_fin_ajuste = time.time()

    tiempo_ajuste = tiempo_fin_ajuste - tiempo_inicio_ajuste
    print(f'Tiempo empleado para el ajuste: {tiempo_ajuste}s')

    #validación cruzada
    tiempo_inicio_validacion_cruzada = time.time()

    score_validacion_cruzada = cross_val_score(
        clasificador,
        x, y,
        scoring = metrica_error,
        cv = k_folds,
        n_jobs = numero_trabajos_paralelos_en_validacion_cruzada
    )
    tiempo_fin_validacion_cruzada = time.time()

    print('score_validacion_cruzada')
    print(score_validacion_cruzada)
    print (f'Media error de validación cruzada {score_validacion_cruzada.mean()}')
    print(f'Varianza del error de validación cruzada: {score_validacion_cruzada.std()}')
 
    print(f'Ein_train {ajuste.score(x,y)}')
    
    print('______Test____')
    print(f'En_test {metrica_error} { ajuste.score(x_test, y_test)}' )
   

    return ajuste




############################################################
############ EVALUACIÓN DE LOS MODELOS #####################
############################################################
ITERACION_MAXIMAS = 2000
# ¿sería interesante ver la variabilidad con los folds ?
k_folds = 5 # valor debe de estar entre 5 y 10



LINEAL_REGRESSION = LinearRegression(normalize = False, 
                    n_jobs = NUMERO_CPUS_PARALELO)

regresion_lineal = Evaluacion(  LINEAL_REGRESSION,
                                x_train_reducido, y_train,
                                x_test_reducido, y_test,
                                k_folds,
                                'Regresión lineal',
                                metrica_error  = 'r2'
                                #metrica_error  = 'neg_mean_squared_error'
                              )


print( 'Máximo coeficiente ', max(regresion_lineal.coef_))

#a = np.array([[3,2],[6,5]])## borrar 
def TransformacionPolinomica( grado,x):
    x_aux = np.copy(x)
    for i in range(1,grado):
        x_aux = x_aux*x_aux
        x = np.c_[x_aux, x]
    return x



grado = 2
regresion_lineal_p2 = Evaluacion(
    LINEAL_REGRESSION,
    TransformacionPolinomica( grado, x_train_reducido),
    y_train,
    TransformacionPolinomica( grado, x_test_reducido,),
    y_test,
    k_folds,
    'Regresión lineal transformación lineal cuadrática',
    metrica_error  = 'r2'
    #metrica_error  = 'neg_mean_squared_error'
)


# No hay mejora considerable, descartamso este método.   
print( 'Máximo coeficiente regresión lineal p2 ', max(regresion_lineal.coef_))
## Número máximo de iteraciones

NUMERO_MAXIMO_ITERACIONES = 5000

##_________ método Ridge ______


RIDGE = Ridge(alpha = 1.0,
              max_iter = NUMERO_MAXIMO_ITERACIONES,
              
              )


ridge =  Evaluacion(  RIDGE,
                      x_train_reducido, y_train,
                      x_test_reducido, y_test,
                      k_folds,
                      'Ridge alpha = 1.0',
                      metrica_error  = 'r2'
                      
                    )

#print(f'Parámetro de ridge: {ridge.coef_}')
print('Máximo parámetro ridge ', max(ridge.coef_))
# La variación es muy poca, y el error en cross validation se mantien, luego descartamso esta opción


#tenemso los datos sufiecientes para aplicar 
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
'''
SUPPORT_VECTOR_REGRESSION = SVR(C=1.0, epsilon=0.2)
grado = 1
svr = Evaluacion(
    SUPPORT_VECTOR_REGRESSION,
    TransformacionPolinomica( grado, x_train_reducido),
    y_train,
    TransformacionPolinomica( grado, x_test_reducido,),
    y_test,
    k_folds,
    'Suppot vector regression',
    metrica_error  = 'r2'
    #metrica_error  = 'neg_mean_squared_error'
)


# datos obtenidos

 Evaluando Regresión lineal transformación lineal cuadrática
Tiempo empleado para el ajuste: 37.32402229309082s
score_validacion_cruzada
[0.32158177 0.25984677 0.31485914 0.28903334 0.32404925]
Media error de validación cruzada 0.3018740548850907
Varianza del error de validación cruzada: 0.02441279456279483
Ein_train 0.3148199169185655
______Test____
En_test r2 0.29736389886545533

Conclusión: no merece la pena
'''

### ______ sgd regresor ________

algoritmos = ['squared_loss', 'epsilon_insensitive']
penalizaciones = ['l1', 'l2'] 
tasa_aprendizaje = ['optimal', 'adaptive']
alphas = [0.001, 0.0001]
eta = 0.0001


cnt = 0 # contado de número de algoritmos lanzados
ajustes = list()

for a in alphas:
    for algoritmo in algoritmos:
        for penalizacion in penalizaciones:
            for aprendizaje in tasa_aprendizaje:
                
                SGD_REGRESSOR = SGDRegressor(
                    alpha = a,
                    max_iter = NUMERO_MAXIMO_ITERACIONES,
                    eta0 = eta,
                    learning_rate = aprendizaje,
                    penalty = penalizacion,
                    loss = algoritmo,
                    shuffle = True,
                    early_stopping = True
                )

                titulo = str(
                    f'\n___SGD regresión ({cnt})___\n' +
                    'algoritmo: ' + algoritmo  + '\n' +
                    'penalización: '+ penalizacion  + '\n' +
                    'aprendizaje: ' +  aprendizaje + '\n' +
                    'eta: ' + str(eta) +  '\n' +
                    'alpha: ' + str(a) + '\n'
                )
                    
                
                sgd =  Evaluacion(  SGD_REGRESSOR,
                                      x_train_reducido, y_train,
                                      x_test_reducido, y_test,
                                      k_folds,
                                      titulo,
                                      metrica_error  = 'r2'
                      
                                  )


                ajustes.append(sgd)
                cnt += 1
