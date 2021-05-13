# Problema de clasificación   

## Análisis del problema   

Se pretende clasificar si el conductor tiene los componentes intactos o defectuosos.   

De la página web de la que se han obtenido los datos ![Dataset for Sensorless Drive Diagnosis Data Set](https://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis#)   


Abstract:   Las características son extraídas de la corriente de un motos. El motor puede tener componentes intactos o defectuosos.  Estos resultados se encuentras en 11 clases con diferentes condiciones.   


Tenemos además la siguiente información:   

- Las características del data set son multivariantes. 
- Los datos son tipo números reales.  
- Es una tarea de clasificación.  
- En número de instancias total es 58509.  
- El número de atributos es de 49.  
- Si faltan datos: N/A .  TO-DO (¿qué hacer si faltan datos)  


## Lectura de los datos   
El fichero tiene extensión `.txt` de texto plano, para leerlo usaremos la biblioteca de `sklearn.datasets.load_files`.   




