import matplotlib.pyplot as plt


def training_model(model, train_data, val_data, epochs=10):
    """
    Executa o treinamento do modelo com os dados de treino/validação e retorna o histórico.
    :param model: 
    :param train_data: 
    :param val_data: 
    :param epochs: 
    :return: 
    """
    print(f"Treinando o modelo por {epochs} épocas...")
    History = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )
    print("Treinamento concluído.")
    return History

def plot_training_history(history, epochs):
    """
    Plota o histórico de treinamento do modelo.
    :param history: 
    :param epochs:
    """
    
    print('Plotando o histórico de treinamento...')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training accuracy')
    plt.plot(epochs_range, val_acc, label='Validation accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training loss')
    plt.plot(epochs_range, val_loss, label='Validation loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.suptitle("Training History")
    plt.show()
