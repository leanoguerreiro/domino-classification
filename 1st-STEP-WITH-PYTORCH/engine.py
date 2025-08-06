import torch

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """
    Treina o modelo por uma época.

    :param model: O modelo a ser treinado.
    :param data_loader: DataLoader contendo os dados de treinamento.
    :param criterion: Função de perda.
    :param optimizer: Otimizador.
    :param device: Dispositivo (CPU ou GPU) para treinamento.
    :return: A perda e a acurácia média da época.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += torch.eq(predicted, labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = float(correct_predictions) / total_samples
    return epoch_loss, epoch_accuracy

def evaluate_model(model, data_loader, criterion, device):
    """
    Avalia o modelo no conjunto de validação.

    :param model: O modelo a ser avaliado.
    :param data_loader: DataLoader contendo os dados de validação.
    :param criterion: Função de perda.
    :param device: Dispositivo (CPU ou GPU) para avaliação.
    :return: A perda e a acurácia média da validação.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += torch.eq(predicted, labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = float(correct_predictions) / total_samples
    return epoch_loss, epoch_accuracy