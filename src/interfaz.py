import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import torch
import torchvision.transforms as transforms
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

mejor_modelo = encontrar_mejor_modelo(ruta_modelo)

print("El modelo que se va a usar es ", mejor_modelo)

mean = [0.4363, 0.4328, 0.3291]
std = [0.2129, 0.2075, 0.2038]
image_resize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            print(file_path)
            prediction = classify(mejor_modelo, image_resize, file_path)
            result_label.config(text=f"Predicción: {classes[prediction]}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar la imagen: {str(e)}")



# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("Clasificación de Imágenes")

choose_button = tk.Button(root, text="Seleccionar Imagen", command=classify_image)
choose_button.pack(pady=20)

result_label = tk.Label(root, text="Predicción: ")
result_label.pack(pady=10)

root.mainloop()


