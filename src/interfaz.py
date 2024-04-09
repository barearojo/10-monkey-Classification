import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import torch
import torchvision.transforms as transforms
from utils import encontrar_mejor_modelo
from functions import classify

classes = [
    "Mantled_howler",
    "Patas_monkey",
    "Bald_uakari",
    "Japanese_macaque",
    "Pygmy_marmoset",
    "White_headed_capuchin",
    "Silvery_marmoset",
    "Common squirrel_monkey",
    "Black_headed_night_monkey",
    "Nilgiri_langur"
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
            image_usuario = Image.open(file_path)
            prediction = classify(mejor_modelo, image_resize, file_path)
            result_label.config(text=f"Predicción: {classes[prediction]}")
            # Mostrar la imagen cargada
            display_image(image_usuario)
            # Mostrar la imagen correspondiente a la clase predicha
            show_class_image(prediction)
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar la imagen: {str(e)}")

def display_image(image):
    # Redimensionar la imagen para ajustarla al tamaño de visualización
    image = image.resize((500, 500), Image.ANTIALIAS)
    # Convertir la imagen a un formato compatible con Tkinter
    img = ImageTk.PhotoImage(image)
    # Actualizar el widget de imagen del usuario
    user_image_label.config(image=img)
    user_image_label.image = img  # Mantener una referencia para evitar que se recolecte el objeto

def show_class_image(prediction):
    # Obtener la ruta de la imagen correspondiente a la clase predicha
    class_image_path = f"./media/monkeys/{classes[prediction]}.jpg"
    print(class_image_path)
    # Verificar si la imagen existe
    if os.path.exists(class_image_path):
        # Abrir y mostrar la imagen correspondiente
        class_image = Image.open(class_image_path)
        # Redimensionar la imagen para ajustarla al tamaño de visualización
        class_image = class_image.resize((500, 500), Image.ANTIALIAS)
        # Convertir la imagen a un formato compatible con Tkinter
        class_img = ImageTk.PhotoImage(class_image)
        # Actualizar el widget de imagen de la clase predicha
        class_image_label.config(image=class_img)
        class_image_label.image = class_img  # Mantener una referencia para evitar que se recolecte el objeto
    else:
        messagebox.showwarning("Advertencia", "No se encontró una imagen correspondiente a esta clase.")

# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("Clasificación de Monos")

choose_button = tk.Button(root, text="Seleccionar Imagen", command=classify_image)
choose_button.grid(row=0, column=0, pady=20, padx=20)

user_image_label = tk.Label(root)
user_image_label.grid(row=1, column=0, padx=20)

class_image_label = tk.Label(root)
class_image_label.grid(row=1, column=1, padx=20)

result_label = tk.Label(root, text="Predicción: ")
result_label.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()

