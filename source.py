# Importa el módulo os para interactuar con el sistema operativo
import os

# Importa el paquete principal de PyTorch
import torch

# Importa torchvision, que proporciona herramientas y conjuntos de datos para la visión por computadora
import torch.utils
import torch.utils.data
import torchvision

# Importa el módulo de transformaciones de torchvision y lo renombra como 'transform'
import torchvision.transforms as transform

import matplotlib.pyplot as plt 
import numpy as np

# Define la ruta al directorio que contiene el conjunto de datos de entrenamiento de imágenes
training_dataset_path = "./data/training/training"
# Define la ruta al directorio que contiene el conjunto de datos de validación de imágenes
training_dataset_path = "./data/validation/validation"


# Define una secuencia de transformaciones para preprocesar las imágenes de entrenamiento
training_resize = transform.Compose([
    # Redimensiona las imágenes a 224x224 píxeles
    transform.Resize((224, 224)),
    # Convierte las imágenes en tensores de PyTorch
    transform.ToTensor()
])

# Crea un objeto ImageFolder que representa el conjunto de datos de imágenes de entrenamiento
# 'root' especifica la ruta al directorio que contiene las imágenes de entrenamiento
# 'transform' aplica las transformaciones definidas a cada imagen durante la carga
train_dataset = torchvision.datasets.ImageFolder(
    root=training_dataset_path,
    transform=training_resize
)

# Crea un DataLoader para el conjunto de datos de entrenamiento
# 'dataset' especifica el conjunto de datos a cargar
# 'batch_size' define el tamaño del lote, es decir, cuántas muestras se cargarán en cada iteración
# 'shuffle' indica si se deben barajar los datos antes de cargarlos en lotes
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)



#El resultado de la función get_mean_std(train_loader) proporciona dos tensores de PyTorch: uno representa la media
#y el otro la desviación estándar de los valores de píxeles en cada canal de color para todas las imágenes del conjunto 
#de datos de entrenamiento. Estos valores son fundamentales para normalizar los datos antes de entrenar un modelo de aprendizaje 
#automático, lo que puede mejorar la convergencia y el rendimiento del modelo al centrar los datos alrededor de cero y escalarlos 
#para tener una varianza uniforme.
# Define una función para calcular la media y la desviación estándar de un conjunto de datos
def get_mean_std(loader):
    # Inicializa la suma de la media y la desviación estándar
    mean = 0
    std = 0
    # Inicializa el contador total de imágenes
    total_images_count = 0
    # Itera sobre los lotes del DataLoader
    for images, _ in loader:
        # Obtiene el tamaño del lote de imágenes
        images_count_batch = images.size(0)
        # Redimensiona las imágenes para que tengan la forma (batch_size, num_channels, num_pixels)
        images = images.view(images_count_batch, images.size(1), -1)
        # Calcula la suma de los valores de píxeles en el eje de los píxeles (dimensión 2) para calcular la media
        mean += images.mean(2).sum(0)
        # Calcula la suma de las desviaciones estándar de los valores de píxeles para calcular la desviación estándar
        std += images.std(2).sum(0)
        # Actualiza el contador total de imágenes
        total_images_count += images_count_batch

    # Calcula la media final dividiendo la suma acumulada por el total de imágenes
    mean /= total_images_count
    # Calcula la desviación estándar final dividiendo la suma acumulada por el total de imágenes
    std /= total_images_count

    return mean, std

# Imprime la media y la desviación estándar del conjunto de datos de entrenamiento
print(get_mean_std(train_loader))

