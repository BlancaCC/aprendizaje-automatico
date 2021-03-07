#!/usr/bin/env python3.8


"""
Autor: Blanca Cano Camarero 
Prática 1 apartado 3
Tarea: 
1.Obtener 100 valores equiespaciados entre 0 y 4π.
2.Obtener el valor de sin(x), cos(x) y tanh(sin(x) + cos(x)) para los 100 valores anteriormente calculados. 
3.Visualizar las tres curvas simultáneamente en el mismo plot (con lı́neas discontinuas en verde, negro y rojo).

"""


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



list(map( lambda t: plt.plot(valores, t[0],t[1]) , zip(imagenes, colores)))

plt.show()
