import timm
import config

def get_model(pretrained: bool = True):
    """
    Cria um modelo usando a biblioteca timm.
    
    A API do timm.create_model permite substituir a camada de classificação
    automaticamente ao passar o argumento `num_classes`.

    Args:
        pretrained (bool): Se True, carrega os pesos pré-treinados.
    
    Returns:
        Um modelo PyTorch.
    """
    model = timm.create_model(
        config.MODEL_NAME,
        pretrained=pretrained,
        num_classes=config.NUM_CLASSES
    )
    
    return model