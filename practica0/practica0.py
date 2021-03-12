#!/usr/bin/env python3.8


"""
Autor: Blanca Cano Camarero 
Prática 1 apartado 1 
Tarea: 
1. Leer la base de datos de iris que hay en scikit-learn. 
2. Obtener las caracterı́sticas (datos de entrada X) y la clase (y). 
3. Quedarse con las caracterı́sticas 1 y 3 (primera y tercera columna de X). 
4. Visualizar con un Scatter Plot los datos, coloreando cada clase con un color diferente (con naranja, negro y verde), e indicando con una leyenda la clase a la que corresponde cada color.

Fuentes: Básicamente la documentación oficial

 https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
 https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html 

  https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
 



"""



import matplotlib.pyplot as plt
from sklearn import datasets

print("---------- Comienza la primera parte  -------- ")

# 1.1
iris = datasets.load_iris()   # leemos la base de datos iris

# 1.2 
print(iris.feature_names) # Con este comando podemos ver el nombre de las caracetristicas
nombre_etiquetas = iris.target_names 
print (nombre_etiquetas) # Nombre de las clases 


X = iris.data # obtenemos los datos
y = iris.target # obtenemos las clases


# 1.3 
X_filtrada = X[::2] # nos quedamos con primera y tercera, que son las pares por eso lo de paso 2



# 1.4

#_____- DIAGRAMA PARA SÉPALO ________
# Calculamos mínimo y máximo de los ejes para representar 
borde = .5
sepalo_longitud_indice = 0
sepalo_ancho_indice  = 1 

x_min, x_max = X[:, sepalo_longitud_indice].min() - borde, X[:, sepalo_longitud_indice].max() + borde
y_min, y_max = X[:, sepalo_ancho_indice ].min() - borde, X[:, sepalo_ancho_indice ].max() + borde

plt.figure("sepalo", figsize=(8, 6)) # identificador y tamaño 
plt.clf() # limpiamos buffer 

# naranja negro verde
colores = ['#ff8000', '#000000', '#009f00']
y_colores = [ colores[i] for i in y] # asociamos a cada clase su color

tam = len(y) // len(nombre_etiquetas) # sabemos que son exactos y ordenados

# mandamos a dibujar cada tipo
[ plt.scatter(X[i*tam:(i+1)*tam, sepalo_longitud_indice], X[i*tam:(i+1)*tam, sepalo_ancho_indice], c=y_colores[i*tam:(i+1)*tam], label=l) for i,l in enumerate(nombre_etiquetas)]

plt.xlabel('Sépalo longitud cm ')
plt.ylabel('Sépalo ancho cm')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend()

plt.show()

#_____- DIAGRAMA PARA PÉTALO ________
# Calculamos mínimo y máximo de los ejes para representar 
borde = .5
petalo_longitud_indice = 2
petalo_ancho_indice  = 3 

x_min, x_max = X[:, petalo_longitud_indice].min() - borde, X[:, petalo_longitud_indice].max() + borde
y_min, y_max = X[:, petalo_ancho_indice ].min() - borde, X[:, petalo_ancho_indice ].max() + borde

plt.figure("petalo", figsize=(8, 6)) # identificador y tamaño 
plt.clf() # limpiamos buffer 

# naranja negro verde 
colores = ['#ff8000', '#000000', '#009f00']
y_colores = [ colores[i] for i in y] # asociamos a cada clase su color


[ plt.scatter(X[i*tam:(i+1)*tam, petalo_longitud_indice], X[i*tam:(i+1)*tam, petalo_ancho_indice], c=y_colores[i*tam:(i+1)*tam], label=l) for i,l in enumerate(nombre_etiquetas)]


plt.xlabel('Pétalo longitud cm ')
plt.ylabel('Pétalo ancho cm')


plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend()

plt.show()


print("---------- fin de la primera parte -------- ")




"""
Autor: Blanca Cano Camarero 
Prática 1 apartado 2 
Tarea: 
Separar en training (75 % de los datos) y test (25 %) aleatoriamente
 conservando la proporción de elementos en cada clase tanto en training
 como en test. 

Con esto se pretende evitar que haya clases infrarepresentadas en entrenamiento o test.
"""


input("Pulse cualquierte tecla para comenzar la corrección de la segunda parte: ")

## Paquetes necesarios: 
#from sklearn import datasets ya usado en apartado 1
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()   # leemos la base de datos iris


numero_clases = len(iris.target_names)


X = iris.data # obtenemos los datos
y = iris.target # obtenemos las clases


test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle = True, stratify=y)

print( 'Datos de entrenamiento: \n' , X_train)
print( 'Target de entrenamiento: \n ', y_train)

print( 'Datos de test: \n' , X_test)
print( 'Target de test: \n ', y_test)

## Comprobación de que la partición está bien hecha

for i in range( len (iris.feature_names)):
    print( 'numero de datos para test de clase ', i )
    print ( sum (  [ 1 if i == i else 0 for i in y_test]))

print( 'Hemos observado que hay la mismas cantidad de datos luego la separación es correcta ')

print("___________ fin de la parte 2 de la práctica ------")






"""
Autor: Blanca Cano Camarero 
Prática 1 apartado 3
Tarea: 
1.Obtener 100 valores equiespaciados entre 0 y 4π.
2.Obtener el valor de sin(x), cos(x) y tanh(sin(x) + cos(x)) para los 100 valores anteriormente calculados. 
3.Visualizar las tres curvas simultáneamente en el mismo plot (con lı́neas discontinuas en verde, negro y rojo).

"""

input("Pulse cualquierte tecla para comenzar la corrección de la tercera parte: ")

# Bibliotecas necesarias para este módulo: 
import numpy as np
import matplotlib.pyplot as plt

# 3.1
limite_sup = 4*np.pi
limite_inf = 0
num_valores = 100

valores = np.linspace(limite_inf, limite_sup, num_valores)

#otra forma alternativa 
#salto = (limite_sup - limite_inf) /num_valores
#valores = np.arange(limite_inf, limite_sup + salto, salto) 

print('Los valores equiespaciados son: \n', valores)


# 3.2 

funciones = [ np.sin, np.cos , lambda x : np.tanh( np.sin(x) + np.cos(x))]

imagenes = [ list(map( f , valores)) for f in funciones ]
print('las respectivas imágenes son ')
list(map( print , imagenes))


# 3.3

#verde, negro y rojo
colores = ["#009f00", "#000000", "#ff0000"]


list(map( lambda t: plt.scatter(valores, t[0], c = [ t[1]]*num_valores) , zip(imagenes, colores)))

plt.show()


print("_______ fin de la corrección _______)
