#!/usr/bin/env python3.8


"""
Autor: Blanca Cano Camarero 
Prática 1 apartado 2 
Tarea: 
Separar en training (75 % de los datos) y test (25 %) aleatoriamente
 conservando la proporción de elementos en cada clase tanto en training
 como en test. 

Con esto se pretende evitar que haya clases infrarepresentadas en entrenamiento o test.
"""

from sklearn import datasets
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
