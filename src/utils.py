import torch  # Importa el módulo torch para operaciones de aprendizaje automático
import os  # Importa el módulo os para interactuar con el sistema operativo
import re  # Importa el módulo re para operaciones con expresiones regulares

def set_device():
    """
    Esta función determina el dispositivo de cálculo disponible y devuelve un objeto que representa ese dispositivo.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Crea un objeto de dispositivo (GPU si está disponible, de lo contrario, CPU)
    return device  # Devuelve el objeto del dispositivo

def encontrar_mejor_modelo(ruta_carpeta):
    """
    Esta función busca el mejor modelo en una carpeta basada en el puntaje del modelo (nombre de archivo).
    :param ruta_carpeta: La ruta de la carpeta donde se encuentran los modelos.
    :return: La ruta completa del mejor modelo encontrado o None si no se encontró ningún modelo.
    """
    mejor_modelo = None  # Inicializa la variable para almacenar el mejor modelo
    mejor_puntaje = 0.0  # Inicializa la variable para almacenar el mejor puntaje

    # Recorre todos los archivos en la carpeta
    for archivo in os.listdir(ruta_carpeta):
        # Utiliza expresiones regulares para buscar el formato del nombre del archivo
        patron = r"mejor_modelo_([\d.]+)\.pth"  # Define el patrón de expresión regular
        coincidencia = re.match(patron, archivo)  # Intenta hacer coincidir el nombre del archivo con el patrón

        if coincidencia:  # Si hay una coincidencia
            puntaje = float(coincidencia.group(1))  # Extrae el puntaje del nombre del archivo coincidente
            if puntaje > mejor_puntaje:  # Si el puntaje es mayor que el mejor puntaje actual
                mejor_puntaje = puntaje  # Actualiza el mejor puntaje
                mejor_modelo = archivo  # Actualiza el mejor modelo

    # Si se encontró un mejor modelo, devuelve la ruta completa
    if mejor_modelo:
        return os.path.join(ruta_carpeta, mejor_modelo)  # Devuelve la ruta completa del mejor modelo
    else:
        return None  # Si no se encontró ningún modelo, devuelve None
