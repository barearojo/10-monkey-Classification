# Importa el módulo os para interactuar con el sistema operativo
import os

# Importa el paquete principal de PyTorch
import torch

# Importa torchvision, que proporciona herramientas y conjuntos de datos para la visión por computadora
import torchvision

# Importa el módulo de transformaciones de torchvision y lo renombra como 'transform'
import torchvision.transforms as transform

# Define la ruta al directorio que contiene el conjunto de datos de entrenamiento de imágenes
training_dataset_path = "./data/training/training"

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

print("aaa")