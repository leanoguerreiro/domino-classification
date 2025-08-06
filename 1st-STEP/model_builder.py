import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from data_augmentation import get_data_augmentation


def build_cnn_model(input_shape: tuple[int, int, int], num_classes: int) -> keras.Model:
    """
    Constrói um modelo de CNN simples para classificação de imagens.
    
    :param input_shape: Tupla representando a forma da entrada (altura, largura, canais).
    :param num_classes: Número de classes para classificação.
    :return: Modelo Keras compilado.
    """
    
    print('Construindo o modelo CNN...')
     
    img_height, img_widht, _ = input_shape
    data_augmentation = get_data_augmentation(img_height, img_widht)
    
    # A arquitetura é um empilhamento linear de camadas convolucionais, seguidas por camadas densas.
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        data_augmentation, # Camada de aumento de dados para aplicar transformações aleatórias nas imagens
        
        layers.Rescaling(1./255), # Normaliza os pixels da imagem para o intervalo [0, 1]
        
        # Camada convolucional com 32 filtros, tamanho de kernel 3x3 e ativação ReLU
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)), # Camada de pooling para reduzir a dimensionalidade
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)), # Outra camada de pooling
        
        layers.Conv2D(64, (3,3), activation='relu'),
        
        layers.Flatten(), # Achata a saída 3D para 1D
        
        layers.Dense(64, activation='relu'),
        
        layers.Dense(num_classes, activation='softmax')
        
    ])
    
    return model

def compile_model(model):
    """
    Compila o modelo com otimizador, função de perda e métricas.
    
    :param model: Modelo Keras a ser compilado.
    :return: Modelo compilado.
    """
    
    print('Compilando o modelo...')
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Usado para classificação multiclasse
        metrics=['accuracy']
    )
    
    return model