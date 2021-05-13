'''
PRÁCTICA 3 Clasificación 
Blanca Cano Camarero   
'''


# Biblioteca lectura de datos   
import pandas as pd


################### funciones auxiliares 
def ReadData (file_name):
    '''
    Input: 
    - file_name: nombre del fichero path relativo a dónde se ejecute 
    La estructura de los datos debe ser: 
       - Cada fila un vector de características con su etiqueta en la última columna.

    Outputs: x,y
    x: matriz de filas de vector de características
    y: vector fila de la etiquetas 
    
    '''

    data = pd.read_csv(file_name,
                       sep = ' ',
                       header = None)
    values = data.values

    # Los datos son todas las filas de todas las columnas salvo la última 
    x = values [:: -1]
    y = values [:, -1] # el vector de características es la últma columana

    return x,y



    
