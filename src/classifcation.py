import os
import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from utils import encontrar_mejor_modelo

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
