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
import torch.nn as red_neuronal
import torch.optim as optim
import torchvision.models as models

# Importa funciones personalizadas de archivos locales
from functions import train_nn  # Importa la función para entrenar la red neuronal
from utils import set_device  # Importa la función para establecer el dispositivo de cálculo

# Define la ruta al directorio que contiene el conjunto de datos de entrenamiento de imágenes
training_dataset_path = "./data/training/training"
# Define la ruta al directorio que contiene el conjunto de datos de validación de imágenes
validation_dataset_path = "./data/validation/validation"

# Define la media y la desviación estándar para normalizar las imágenes
mean = [0.4363, 0.4328, 0.3291]
std = [0.2129, 0.2075, 0.2038]

# Define una secuencia de transformaciones para preprocesar las imágenes de entrenamiento
training_resize = transform.Compose([
    # Redimensiona las imágenes a 224x224 píxeles (PUEDE AFECTAR A LA PERFORMANCE DEL MODELO)
    transform.Resize((224, 224)),
    transform.RandomHorizontalFlip(),  # Voltea aleatoriamente horizontalmente las imágenes
    transform.RandomRotation(20),  # Rota aleatoriamente las imágenes en el rango de -20 a +20 grados
    # Convierte las imágenes en tensores de PyTorch
    transform.ToTensor(),
    # Normaliza las imágenes con la media y desviación estándar especificadas
    transform.Normalize(torch.Tensor(mean),torch.Tensor(std))
])

# Define una secuencia de transformaciones para preprocesar las imágenes de validación
validation_resize = transform.Compose([
    # Redimensiona las imágenes a 224x224 píxeles (PUEDE AFECTAR A LA PERFORMANCE DEL MODELO)
    transform.Resize((224, 224)),
    # Convierte las imágenes en tensores de PyTorch
    transform.ToTensor(),
    # Normaliza las imágenes con la media y desviación estándar especificadas
    transform.Normalize(torch.Tensor(mean),torch.Tensor(std))
])

# Crea datasets de entrenamiento y validación con las transformaciones definidas
train_dataset = torchvision.datasets.ImageFolder(root=training_dataset_path, transform=training_resize)
validation_dataset = torchvision.datasets.ImageFolder(root=validation_dataset_path, transform=validation_resize)

# Crea loaders de entrenamiento y validación para facilitar la carga de datos
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=32, shuffle=False)

# Llama a la función set_device para obtener el dispositivo de cálculo (CPU o GPU)
device = set_device()
print("Para entrenar la red neuronal se va a utilizar el siguiente componente ",device)

# Selecciona un modelo pre-entrenado de ResNet-18
model_usado = models.resnet18(pretrained=True)

# Obtiene el número de características de la capa de clasificación
num_ftr = model_usado.fc.in_features

# Define el número de clases para la clasificación (en este caso, 10 clases)
number_classes = 10

# Modifica la capa de clasificación para adaptarla al número de clases en el conjunto de datos
model_usado.fc = red_neuronal.Linear(num_ftr, number_classes)

# Mueve el modelo a la GPU si está disponible
model_con_device = model_usado.to(device)

# Define la función de pérdida para el entrenamiento (entropía cruzada)
loss_fn = red_neuronal.CrossEntropyLoss()

# Define el optimizador para el entrenamiento (SGD con momentum)
optimizer = optim.SGD(model_con_device.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

# Entrena la red neuronal con los datos de entrenamiento y validación
train_nn(model_con_device, train_loader, validation_loader, loss_fn, optimizer, 15)  # Entrena durante 15 épocas

# Carga el mejor modelo guardado según el checkpoint
checkpoint = torch.load('./models/model_best_checkpoint.pth.tar')

# Imprime la información del mejor modelo guardado
print("La mejor iteración fue ", checkpoint['epoch'], "con una precisión de ", checkpoint['best_accuracy'])

# Carga los pesos del mejor modelo guardado en el modelo actual
model_con_device.load_state_dict(checkpoint['model'])

# Obtiene la precisión del mejor modelo guardado
best_accuracy = checkpoint['best_accuracy']

# Guarda el mejor modelo con su precisión en el nombre del archivo
torch.save(model_con_device, f"./models/mejor_modelo_{best_accuracy:.2f}.pth")







