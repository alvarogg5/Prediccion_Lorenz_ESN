# Descripción del Proyecto

Este código (`ÈSN_Lorenz.py`) implementa una Echo State Network (ESN) para predecir datos del sistema de Lorenz. La ESN utiliza una matriz generada aleatoriamente y datos de entrada para entrenar y realizar predicciones. Proporciona visualizaciones para comparar las predicciones con los datos reales y muestra el error de predicción normalizado.

## Requisitos

- Python 3.x
- NumPy
- Matplotlib
- Pandas

## Cómo utilizar el proyecto

1. Asegúrate de tener instalados los requisitos mencionados anteriormente.
2. Descarga el archivo `u_csv10000.csv` y colócalo en el mismo directorio que el archivo Python.
3. Importa las bibliotecas requeridas y el código proporcionado en tu entorno de desarrollo de Python.


4. Utiliza la función `repository_onlyL` para realizar la predicción. Puedes modificar los parámetros dentro de esta función según tus necesidades. Por ejemplo:

```python
repository_onlyL(M=3, T=500, nT=2, Dr=800, d=3, rho=0.45, sigma=0.2, beta=1.4)
```

5. La función `repository_onlyL` generará una predicción y mostrará un gráfico 3D que compara los datos reales y la predicción.

## Funciones principales

### A_build(width, p, rho)

Esta función genera una matriz `A` con una probabilidad `p` de tener conexiones entre sus elementos y una densidad espectral `rho`.

### win_build(width, height, sigma)

Esta función crea una matriz `W_in` que se utiliza como entrada en la predicción. La matriz tiene dimensiones `width` x `height` y contiene valores aleatorios dentro del rango [-sigma, sigma].

### repository_onlyL(M, T, nT, Dr, d, rho, sigma, beta)

Esta función es la función principal del proyecto. Utiliza las matrices generadas por `A_build` y `win_build` junto con datos reales cargados desde el archivo `u_csv10000.csv` para realizar la predicción. Muestra gráficos que comparan los datos reales con la predicción y devuelve el tiempo de predicción utilizable.

## Notas adicionales

- Si deseas ajustar los parámetros o probar diferentes configuraciones, puedes hacerlo modificando los valores proporcionados en la función `repository_onlyL`.

- Puedes descomentar y modificar la sección que contiene el gráfico de "Usable Time vs. rho" para explorar el efecto de diferentes valores de `rho` en el tiempo de predicción utilizable.

Recuerda que este README es solo una guía general sobre cómo utilizar el proyecto. Asegúrate de que tu entorno de desarrollo esté configurado correctamente y de tener los archivos y bibliotecas necesarios antes de ejecutar el código. ¡Diviértete explorando y experimentando con el proyecto!

#Predicción de Variables Meteorológicas utilizando Redes Neuronales Recurrentes (ESN)

Este código (`ÈSN_meteo.py`) ha sido modificado para predecir variables meteorológicas, específicamente, se ha implementado la predicción de tres variables: lluvia, racha máxima y temperatura media. 

Se realizan varias transformaciones y manipulaciones de los datos, como el preprocesamiento para eliminar valores no válidos y el ajuste de tipos de datos. Luego, la predicción se lleva a cabo y se muestran gráficos comparando los datos reales con los valores predichos.

La función repository_onlyL es la función principal que realiza el entrenamiento y la predicción. Se han agregado gráficos adicionales para cada una de las variables meteorológicas, lo que permite una visualización más detallada de las predicciones. Además, se ha ajustado el rango de tiempo en el que se realizan las predicciones.

En general, este código representa una extensión significativa del proyecto original, centrado ahora en la predicción de variables meteorológicas. Asegúrate de tener descargado el archivo `2014-2022a.csv`, que contiene los datos meteorológicos, antes de ejecutar el código.
