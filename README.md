# Clasificación de Imágenes de Monos

Este repositorio contiene un conjunto de scripts para clasificar imágenes de diferentes especies de monos utilizando una red neuronal pre-entrenada. A continuación se describen los diferentes scripts y su funcionalidad.

## Requisitos

Es necesario tener instalado Python 3.x junto con las siguientes bibliotecas:

- torch
- torchvision
- tkinter
- PIL
- matplotlib
- numpy

Puede instalar estas dependencias utilizando pip:

pip install torch torchvision tkinter matplotlib numpy


## Información Adicional

Es necesario tener en cuenta que este código está adaptado tanto para si se posee una tarjeta gráfica que pueda usar CUDA como una que no. Se recomienda utilizar en máquinas donde se pueda utilizar CUDA, ya que así fue entrenada la red neuronal para este proyecto.

## Instrucciones de Uso

1. Ejecute `python3 src/std_mean.py` (solo necesario si se cambian los datos de entrenamiento) para calcular la media y la desviación estándar del conjunto de datos de entrenamiento.
2. Copie los resultados obtenidos y póngalos dentro de `train.py`.
3. Ejecute `python3 src/train.py` (solo necesario si se quiere volver a entrenar la red neuronal, es necesario cambiar en el código el número de iteraciones).
4. Ejecute `python3 src/interfaz.py` para iniciar la interfaz gráfica de clasificación de imágenes de monos.

## Capturas de Funcionamiento


## Descripción de los Scripts

1. `train.py`: Script para entrenar la red neuronal utilizando los datos de entrenamiento proporcionados. También se encarga de guardar el mejor modelo obtenido durante el entrenamiento.
2. `interfaz.py`: Script para ejecutar una interfaz gráfica que permite al usuario cargar una imagen y clasificarla utilizando el modelo pre-entrenado.
3. `std_mean.py`: Script para calcular la media y la desviación estándar del conjunto de datos de entrenamiento, necesarios para la normalización de datos durante el entrenamiento.
4. `functions.py`: Contiene funciones auxiliares utilizadas en los otros scripts, como la función para clasificar una imagen y la función para encontrar el mejor modelo en una carpeta.
5. `utils.py`: Contiene funciones de utilidad, como la función para establecer el dispositivo de cálculo (CPU o GPU) y la función para guardar un punto de control del modelo durante el entrenamiento.

## Autor

Este proyecto fue desarrollado por [Nombre del Autor].

---
Este README fue generado automáticamente.
