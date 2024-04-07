import torch
import os
import re

def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def encontrar_mejor_modelo(ruta_carpeta):
    mejor_modelo = None
    mejor_puntaje = 0.0

    # Recorre todos los archivos en la carpeta
    for archivo in os.listdir(ruta_carpeta):
        # Utiliza expresiones regulares para buscar el formato del nombre del archivo
        patron = r"mejor_modelo_([\d.]+)\.pth"
        coincidencia = re.match(patron, archivo)
        if coincidencia:
            puntaje = float(coincidencia.group(1))
            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_modelo = archivo

    # Si se encontró un mejor modelo, devuelve la ruta completa
    if mejor_modelo:
        return os.path.join(ruta_carpeta, mejor_modelo)
    else:
        return None