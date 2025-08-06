from tensorflow import keras
from tensorflow.keras import layers

def get_data_augmentation(img_height, img_width) -> keras.Sequential:
    """
    Cria um pipeline de aumento de dados para imagens.
    
    :return: Um modelo Keras Sequential contendo camadas de aumento de dados.
    """
    
    print('Criando o pipeline de aumento de dados...')
    
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"), # Apenas flip horizontal
        layers.RandomRotation(0.1),      # Rotação reduzida
        layers.RandomZoom(0.1),        # Zoom reduzido
        
        
    ], name='data_augmentation_pipeline'
    )
    # As seguintes camadas estão comentadas, mas podem ser descomentadas para aplicar mais aumentos de dados
    """
    layers.RandomTranslation(0.1, 0.1), # Translada aleatoriamente as imagens em até 10% na altura e largura
    layers.GaussianNoise(0.1, 123), # Adiciona ruído gaussiano com desvio padrão de 0.1 e semente 123
    layers.RandomGrayscale(0.2), # Converte aleatoriamente 20% das imagens para escala de cinza
    layers.RandomContrast(0.2), # Aplica contraste aleatório com fator de 0.2
    layers.RandomBrightness(0.2) # Aplica brilho aleatório
    """
    return data_augmentation