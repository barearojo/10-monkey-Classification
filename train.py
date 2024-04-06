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

from red_neuronal import train_nn

# Define la ruta al directorio que contiene el conjunto de datos de entrenamiento de imágenes
training_dataset_path = "./data/training/training"
# Define la ruta al directorio que contiene el conjunto de datos de validación de imágenes
validation_dataset_path = "./data/validation/validation"


mean = [0.4363, 0.4328, 0.3291]
std = [0.2129, 0.2075, 0.2038]


# Define una secuencia de transformaciones para preprocesar las imágenes de entrenamiento
training_resize = transform.Compose([
    # Redimensiona las imágenes a 224x224 píxeles PUEDE AFECTAR A LA PERFORMANCE DEL MODELO
    transform.Resize((224, 224)),
    transform.RandomHorizontalFlip(),
    transform.RandomRotation(20),
    # Convierte las imágenes en tensores de PyTorch
    transform.ToTensor(),
    #normalización
    transform.Normalize(torch.Tensor(mean),torch.Tensor(std))
])


# Define una secuencia de transformaciones para preprocesar las imágenes de validación
validation_resize = transform.Compose([
    # Redimensiona las imágenes a 224x224 píxeles PUEDE AFECTAR A LA PERFORMANCE DEL MODELO
    transform.Resize((224, 224)),
    # Convierte las imágenes en tensores de PyTorch
    transform.ToTensor(),
    #normalización
    transform.Normalize(torch.Tensor(mean),torch.Tensor(std))
])



train_dataset = torchvision.datasets.ImageFolder( root=training_dataset_path, transform=training_resize)
validation_dataset = torchvision.datasets.ImageFolder(root= validation_dataset_path, transform=validation_resize)

#empezar con 32 y luego ir subiendo multiplicanod por dos hasta encontrar el mejor
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=32, shuffle=False)

def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


# Llamando a la función set_device para obtener el dispositivo
device = set_device()
print("Para entrenar la red neuronal se va a utilizar el siguiente componente ",device)

#se ha elegido el 18 debido a que ofrece un buen balance entre eficiencia y precisión
model_usado = models.resnet18(pretrained=True)


#numero de caracteristicas a tener en cuenta
num_ftr = model_usado.fc.in_features

#numero de distintas clases 
number_classes = 10

#prepara la matrices de propagración
model_usado.fc = red_neuronal.Linear(num_ftr, number_classes)

#le añadimos el device que vamos a usar
model_con_device = model_usado.to(device)

#funcion de perdida
loss_fn = red_neuronal.CrossEntropyLoss()

#elegimos optimizador
optimizer = optim.SGD(model_con_device.parameters(), lr = 0.01, momentum=0.9, weight_decay=0.003)

train_nn(model_con_device,train_loader,validation_loader,loss_fn,optimizer,20)




