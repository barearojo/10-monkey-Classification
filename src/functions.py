import torch
from utils import set_device
from PIL import Image
import torchvision

def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    """
    Función para entrenar un modelo de red neuronal.

    Args:
        model (torch.nn.Module): El modelo PyTorch a entrenar.
        train_loader (torch.utils.data.DataLoader): DataLoader para el conjunto de datos de entrenamiento.
        test_loader (torch.utils.data.DataLoader): DataLoader para el conjunto de datos de prueba.
        criterion: La función de pérdida utilizada para calcular el error del modelo.
        optimizer (torch.optim.Optimizer): El optimizador utilizado para entrenar el modelo.
        n_epochs (int): El número de épocas de entrenamiento.

    Returns:
        torch.nn.Module: El modelo entrenado.
    """
    device = set_device()  # Establece el dispositivo de ejecución (CPU o GPU).
    best_acc = 0  # Inicializa la mejor precisión en cero.

    for epoch in range(n_epochs):  # Itera sobre el número de épocas especificado.
        print("Iteración número", (epoch + 1))  # Imprime el número de época actual.
        model.train()  # Establece el modelo en modo de entrenamiento, activando la normalización de dropout y batch.
        running_loss = 0.0  # Inicializa la pérdida acumulada en cero.
        running_correct = 0.0  # Inicializa el número de predicciones correctas acumuladas en cero.
        total = 0  # Inicializa el contador total de imágenes evaluadas en cero.

        for data in train_loader:  # Itera sobre el conjunto de datos de entrenamiento.
            images, labels = data  # Obtiene las imágenes y las etiquetas del lote actual.
            images = images.to(device)  # Transfiere las imágenes al dispositivo de ejecución.
            labels = labels.to(device)  # Transfiere las etiquetas al dispositivo de ejecución.
            total += labels.size(0)  # Actualiza el contador total de imágenes.

            optimizer.zero_grad()  # Reinicia los gradientes del optimizador.

            outputs = model(images)  # Realiza una inferencia con el modelo.

            _, predicted = torch.max(outputs.data, 1)  # Obtiene las predicciones del modelo.

            loss = criterion(outputs, labels)  # Calcula la pérdida del modelo.

            loss.backward()  # Realiza la retropropagación del error.

            optimizer.step()  # Realiza un paso de optimización.

            running_loss += loss.item()  # Actualiza la pérdida acumulada.
            running_correct += (labels == predicted).sum().item()  # Actualiza el número de predicciones correctas acumuladas.

        epoch_loss = running_loss / len(train_loader)  # Calcula la pérdida promedio por época.
        epoch_acc = 100.00 * running_correct / total  # Calcula la precisión del modelo en el conjunto de datos de entrenamiento.

        print("----- Training dataset got {} out of {} images correct ({}%). Con una pérdida de {}".format(running_correct, total, epoch_acc, epoch_loss))  # Imprime la precisión y la pérdida del modelo en el conjunto de datos de entrenamiento.

        test_acc = evaluate_model(model, test_loader, optimizer)  # Evalúa el modelo en el conjunto de datos de prueba.
        if test_acc > best_acc:  # Comprueba si la precisión en el conjunto de datos de prueba es mejor que la mejor precisión registrada hasta ahora.
            best_acc = test_acc  # Actualiza la mejor precisión.
            save_checkpoint(model, epoch, optimizer, best_acc)  # Guarda un punto de control del modelo.

    print("Entrenamiento terminado")  # Indica que el entrenamiento ha finalizado.
    return model  # Devuelve el modelo entrenado.



def evaluate_model(model, validation_loader, optimizer):
    """
    Función para evaluar el modelo en un conjunto de datos de validación.

    Args:
        model (torch.nn.Module): El modelo PyTorch a evaluar.
        validation_loader (torch.utils.data.DataLoader): DataLoader para el conjunto de datos de validación.
        optimizer (torch.optim.Optimizer): El optimizador utilizado para entrenar el modelo.

    Returns:
        float: La precisión del modelo en el conjunto de datos de validación.
    """
    model.eval()  # Establece el modelo en modo de evaluación, desactivando la normalización de dropout y batch.
    predicted_correct_epoch = 0  # Inicializa el contador de predicciones correctas en la época.
    total = 0  # Inicializa el contador total de imágenes evaluadas.
    device = set_device()  # Establece el dispositivo de ejecución (CPU o GPU).

    with torch.no_grad():  # Desactiva el cálculo de gradientes durante la evaluación.
        for data in validation_loader:  # Itera sobre el conjunto de datos de validación.
            images, labels = data  # Obtiene las imágenes y las etiquetas del lote actual.
            images = images.to(device)  # Transfiere las imágenes al dispositivo de ejecución.
            labels = labels.to(device)  # Transfiere las etiquetas al dispositivo de ejecución.
            total += labels.size(0)  # Actualiza el contador total de imágenes.

            optimizer.zero_grad()  # Reinicia los gradientes del optimizador.

            outputs = model(images)  # Realiza una inferencia con el modelo.

            _, predicted = torch.max(outputs.data, 1)  # Obtiene las predicciones del modelo.
            predicted_correct_epoch += (labels == predicted).sum().item()  # Actualiza el contador de predicciones correctas.

    epoch_acc = 100.00 * predicted_correct_epoch / total  # Calcula la precisión del modelo en el conjunto de datos de validación.
    print("----- Testing dataset got {} out of {} images correct ({}%)".format(predicted_correct_epoch, total, epoch_acc))  # Imprime la precisión del modelo.
    return epoch_acc  # Devuelve la precisión del modelo en el conjunto de datos de validación.



def save_checkpoint(model, epoch, optimizer, best_acc):
    """
    Función para guardar un punto de control del modelo.

    Args:
        model (torch.nn.Module): El modelo PyTorch.
        epoch (int): El número de época actual.
        optimizer (torch.optim.Optimizer): El optimizador utilizado para entrenar el modelo.
        best_acc (float): La mejor precisión alcanzada durante el entrenamiento.
    """
    # Crea un diccionario 'state' que contiene el estado actual del modelo, época, mejor precisión y estado del optimizador.
    state = {
        'model': model.state_dict(),  # Guarda el estado del modelo
        'epoch': epoch + 1,  # Guarda el número de época actual aumentado en 1
        'best_accuracy': best_acc,  # Guarda la mejor precisión alcanzada durante el entrenamiento
        'optimizer': optimizer.state_dict(),  # Guarda el estado del optimizador
    }
    # Define el nombre del archivo donde se guardará el punto de control
    filename = f"./models/model_best_checkpoint.pth.tar"
    # Guarda el diccionario 'state' en un archivo usando torch.save()
    torch.save(state, filename)


def classify(model_path, image_transforms, image_path):
    # Cargar el modelo desde el archivo .pth
    model = torch.load(model_path)
    # Establecer el modelo en modo de evaluación
    model = model.eval()

    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    return predicted.item()