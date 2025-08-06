from data_utils import load_and_process_data
from model_builder import build_cnn_model, compile_model
from training_utils import training_model, plot_training_history

if __name__ == '__main__':
    DATASET_PATH = "../data/data-raw"
    IMG_HEIGTH = 100
    IMG_WIDTH = 100
    BATCH_SIZE = 32
    EPOCHS = 20
    
    # Carregar os dados de treinamento e validação
    train_data, val_data, class_names = load_and_process_data(
        dataset_dir=DATASET_PATH,
        img_size=(IMG_HEIGTH, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    
    # Construir o Modelo CNN
    
    model = build_cnn_model(
        input_shape=(IMG_HEIGTH, IMG_WIDTH, 3),
        num_classes=len(class_names)
    )
    
    # Compilar o modelo
    model = compile_model(model)
    
    # Descomentar caso queira visualizar a arquitetura do modelo
    #model.summary() # Exibir arquitetura do modelo
    
    training_history = training_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=EPOCHS,
    )
    
    # Plotar o histórico de treinamento
    plot_training_history(training_history, epochs=EPOCHS)