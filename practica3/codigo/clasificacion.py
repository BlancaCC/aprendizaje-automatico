'''
PRÁCTICA 3 Clasificación 
Blanca Cano Camarero   
'''
#############################
#######  BIBLIOTECAS  #######
#############################
# Biblioteca lectura de datos   
import pandas as pd

# matemáticas
# ==========================
import numpy as np


# Preprocesado y modelado
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

###############################



########## CONSTANTES #########
NOMBRE_FICHERO_CLASIFICACION = './datos/Sensorless_drive_diagnosis.txt'
SEPARADOR_CLASIFICACION = ' '

################### funciones auxiliares 
def LeerDatos (nombre_fichero, separador):
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
                       header = None)
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


###########################################################
#### Herramientas básicas


def ExploracionInicial(x):

    media = x.mean(axis = 0)
    varianza = x.var(axis = 0)
    
    print('Exploración inicial datos: \n')
    
    print('\nMedia de cada variable')
    print(media)


    print('\nVarianza ')
    print(varianza)


    print('-'*20)
    print('Resumen de las tablas')
    print('-'*20)
    
    print('\nMedia')
   
    print(f'Valor mínimo de las medias {min(media)}')
    print(f'Valor máximo de las medias {max(media)}')
    print('\nVarianza ')
   
    print(f'Valor mínimo de las varianzas {min(varianza)}')
    print(f'Valor máximo de las varianzas {max(varianza)}')
    
    print('-'*20)

    

        
###########################################################
###########################################################
###########################################################
print(f'Procedemos a leer los datos del fichero {NOMBRE_FICHERO_CLASIFICACION}')
x,y = LeerDatos( NOMBRE_FICHERO_CLASIFICACION, SEPARADOR_CLASIFICACION)

ExploracionInicial(x)

''' # COMENTO PORQUE TARDA MUCHO LA EJECUCIÓN   
print('PCA con escalado de datos')
pca_pipe = make_pipeline(StandardScaler(), PCA())
pca_pipe.fit(x)

# Se extrae el modelo entrenado del pipeline
modelo_pca = pca_pipe.named_steps['pca']
''
print('Vamos a representar los datos usando el algoritmo TSNE, este tarda un par de minutos')
x_tsne = TSNE(n_components=2).fit_transform(modelo_pca.components._modelo_pca.components_T)



tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(x)
print('t-SNE done! Time elapsed: ')

VisualizarClasificacion2D(tsne_results, y)
Separador('fin de la visualización')
'''

### Comprobación si los datos están balanceados   
def NumeroDeEtiquetas(y):
    '''
    INPUT: y: etiquetas 
    OUTPUT conteo: diccionario que asocia cada etiqueta con el número de veces que aparece 
    '''
    conteo = dict()
    etiquetas_unicas = np.unique(y)
    
    for i in etiquetas_unicas:
        conteo [i] = np.sum( y == i)
    return conteo

def ImprimeDiccionario(diccionario, titulos):

    print( ' | '.join(titulos) + '  \t  ')
    print ('--- | ' * (len(titulos)-1) + '---    ')
    for k,v in diccionario.items():
        print(k , ' | ', v , '    ')
        

print('Comprobación de balanceo')
ImprimeDiccionario(
    NumeroDeEtiquetas(y),
    ['Etiqueta', 'Número apariciones'])

Separador('Separamos test y entrenamiento')

###### separación test y entrenamiento  #####
ratio_test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size= ratio_test_size,
    shuffle = True, 
    random_state=1)

print('Veamos si ha sido homogéneo')

ImprimeDiccionario(
    NumeroDeEtiquetas(y_test),
    ['Etiqueta', 'Número apariciones'])

Separador('Normalización')  

print('Datos sin normalizar ')
ExploracionInicial(X_train)

print ('Datos normalizados')

## Normalización de los datos
'''
scaler = StandardScaler()
train_X = scaler.fit_transform( train_X )
test_X = scaler.transform( test_X )
'''
