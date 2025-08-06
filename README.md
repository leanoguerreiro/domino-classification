# Domino Classification 🀄

Este repositório contém dois pipelines completos para classificação de peças de dominó a partir de imagens, implementados tanto em **TensorFlow/Keras** quanto em **PyTorch**.  
O objetivo é treinar modelos de visão computacional capazes de identificar o valor das peças de dominó a partir de fotos.

## 📂 Estrutura do Projeto

```
leanoguerreiro-domino-classification/
│
├── 1st-STEP/                 # Versão com TensorFlow/Keras
│   ├── data_augmentation.py  # Pipeline de aumento de dados
│   ├── data_utils.py         # Funções para carregamento e pré-processamento dos dados
│   ├── main.py               # Script principal de treino
│   ├── model_builder.py      # Construção e compilação do modelo CNN
│   └── training_utils.py     # Funções de treino e plotagem de métricas
│
├── 1st-STEP-WITH-PYTORCH/    # Versão com PyTorch
│   ├── config.py             # Configurações globais do treino
│   ├── dataset.py            # Transformações e carregamento de dados
│   ├── engine.py             # Funções de treino e avaliação
│   ├── model.py              # Criação do modelo (usando timm)
│   └── train.py              # Treinamento com validação cruzada (K-Fold)
│
└── README.md                 # Documentação do projeto
```

## 🚀 Funcionalidades

- **Dois frameworks**: escolha entre TensorFlow/Keras ou PyTorch.
- **Data Augmentation**: flips, rotações, zoom, normalização e outros.
- **Modelos personalizáveis**:
  - TensorFlow: CNN simples feita do zero.
  - PyTorch: Modelos da biblioteca `timm` (ex.: ResNet50) com pesos pré-treinados no ImageNet.
- **Validação cruzada (K-Fold)** para avaliação mais robusta.
- **Treinamento final** com todo o conjunto de dados.
- **Salvamento de modelo treinado**.

## 📦 Pré-requisitos

Antes de rodar o projeto, instale as dependências:

```bash
pip install -r requirements.txt
```

Dependências principais:
- TensorFlow
- PyTorch + torchvision
- timm
- scikit-learn
- matplotlib
- numpy

## 📊 Dataset

Coloque as imagens na pasta:

```
data/data-raw/
    ├── classe_1/
    │    ├── img1.jpg
    │    ├── img2.jpg
    ├── classe_2/
    │    ├── img1.jpg
    │    ├── img2.jpg
    ...
```

Cada subpasta representa uma **classe** (ex.: valor da peça de dominó).

## 🏃 Como Executar

### 1️⃣ Versão TensorFlow/Keras
```bash
cd 1st-STEP
python main.py
```

O script:
- Carrega e divide o dataset.
- Constrói uma CNN.
- Treina e exibe métricas.
- Plota a acurácia e perda de treino/validação.

### 2️⃣ Versão PyTorch
```bash
cd 1st-STEP-WITH-PYTORCH
python train.py
```

O script:
- Executa validação cruzada K-Fold.
- Exibe acurácia por fold e média final.
- Treina o modelo final com todos os dados.
- Salva o modelo treinado em `./models/`.

## ⚙️ Configuração

Ajuste os hiperparâmetros no arquivo `config.py` (na versão PyTorch) ou no `main.py` (na versão TensorFlow):

```python
MODEL_NAME = "resnet50"   # Nome do modelo no timm
NUM_CLASSES = 28          # Número de classes
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
```

## 📈 Resultados Esperados

- **TensorFlow**: Gráficos de acurácia e perda.
- **PyTorch**: Métricas de cada fold e média final.
- Modelo treinado salvo para uso posterior.

## 📜 Licença

Este projeto é distribuído sob a licença MIT. Sinta-se livre para usar e modificar.

---

💡 **Dica:**  
Se você quiser apenas treinar rapidamente, pode usar a versão TensorFlow.  
Para uma avaliação mais robusta e uso de arquiteturas modernas pré-treinadas, opte pela versão PyTorch.
