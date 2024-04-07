import torch
from utils import set_device

def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    device = set_device()
    best_acc = 0

    for epoch in range(n_epochs):
        print("Iteración número", (epoch + 1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(images)

            _, predicted = torch.max(outputs.data,1)

            loss = criterion(outputs,labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00 *  running_correct/ total
        print("----- Training dataset got {} out of {} images correct ({}%). Con una pérdida de {}".format(running_correct, total, epoch_acc, epoch_loss))

        test_acc = evaluate_model(model,test_loader,optimizer)
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, epoch, optimizer, best_acc)
    
    print("Entrenamiento terminado")
    return model


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
