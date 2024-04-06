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

        test_acc = evaluate_model(model,test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, epoch, optimizer, best_acc)
    
    print("Entrenamiento terminado")
    return model


def evaluate_model(model, test_loader):
    model.eval()
    predicted_correct_epoch = 0
    total = 0
    device = set_device()

    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(images)

            _, predicted = torch.max(outputs.data,1)
            predicted_correct_epoch += (labels==predicted).sum().item()

    epoch_acc = 100.00 *  predicted_correct_epoch/ total
    print("----- Testing dataset got {} out of {} images correct ({}%)".format(predicted_correct_epoch, total, epoch_acc))
    return epoch_acc

def save_checkpoint(model, epoch, optimizer, best_acc):
    state = {
        'model' : model.state_dict(),
        'epoch' : epoch + 1,
        'best_accuracy' : best_acc,
        'optimizer' : optimizer.state_dict(),
    }
    torch.save(state, 'model_best_checkpoint.pth.tar')