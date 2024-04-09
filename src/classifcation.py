import os
import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from utils import encontrar_mejor_modelo
from functions import classify

classes = [
    "Mantled howler",
    "Patas monkey",
    "Bald uakari",
    "Japanese macaque",
    "Pygmy marmoset",
    "White headed capuchin",
    "Silvery marmoset",
    "Common squirrel monkey",
    "Black headed night monkey",
    "Nilgiri langur"
]

ruta_modelo = "./models"

mejor_modelo  = encontrar_mejor_modelo(ruta_modelo)

print("El modelo que se va a usar es ", mejor_modelo)


mean = [0.4363, 0.4328, 0.3291]
std = [0.2129, 0.2075, 0.2038]
image_resize = transforms.Compose([
    # Redimensiona las imágenes a 224x224 píxeles PUEDE AFECTAR A LA PERFORMANCE DEL MODELO
    transforms.Resize((224, 224)),
    # Convierte las imágenes en tensores de PyTorch
    transforms.ToTensor(),
    #normalización
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])

prediccion = classify(mejor_modelo, image_resize,"/home/juan/proyectos/10-monkey-Classification/mantled-howler.jpg")
print(classes[prediccion])
