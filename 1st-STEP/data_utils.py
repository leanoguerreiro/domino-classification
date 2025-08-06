import tensorflow as tf


def load_and_process_data(dataset_dir:str, img_size: tuple[int, int], batch_size:int, validation_split=0.2):
    """
    Carrega as imagens do diretório, divide em treino e validação e retorna os datasets
    
    :param dataset_dir: 
    :param img_size: 
    :param batch_size: 
    :param validation_split: 
    :return: 
    """
    
    print("[INFO] Carregando imagens...")
    
    # Carrega o conjunto de dados de treino a partir do diretório
    train_data = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        verbose=True
    )    
    
    validation_data = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        verbose=True
    )
    
    class_names = train_data.class_names
    print(f"Módulo de Dados: Encontradas {len(class_names)} classes.")    
    
    # Otimiza o carregamento de dados para melhor desempenho
    autotune = tf.data.AUTOTUNE
    train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=autotune)
    validation_data = validation_data.cache().prefetch(buffer_size=autotune)
    
    return train_data, validation_data