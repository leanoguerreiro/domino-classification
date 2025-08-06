import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import numpy as np

import config
from model import get_model
from dataset import get_dataset, data_transforms
from engine import train_one_epoch, evaluate_model

def run_training():
    """
    Função principal para treinar o modelo de classificação de imagens.
    """
    print(f"Definindo o dispositivo: {config.DEVICE}")
    print(f"Definindo o modelo de imagens: {config.MODEL_NAME}")
    
    dataset = get_dataset()
    targets = dataset.targets
    kfold = StratifiedKFold(
        n_splits=config.K_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED
    )
    fold_results = []
    
    for fold, (train_index, val_index) in enumerate(kfold.split(dataset, targets)):
        print(f'\n{"-"*20} FOLD {fold+1}/{config.K_FOLDS} {"-"*20}')
        
        train_sampler = SubsetRandomSampler(train_index)
        val_sampler = SubsetRandomSampler(val_index)
        
        dataset.transforms = data_transforms['train']
        train_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True
            
        )
        
        dataset.transforms = data_transforms['val']
        val_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            sampler=val_sampler,
            num_workers=8,
            pin_memory=True
        )
        
        model = get_model(pretrained=True).to(config.DEVICE)
        optimizer = optim.Adam(
            model.get_classifier().parameters(),
            lr=config.LEARNING_RATE
        )
        criterion = nn.CrossEntropyLoss() 
        
        best_acc = 0.0
        
        for epoch in range(config.NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, config.DEVICE)

            print(f'Época {epoch+1:02d}/{config.NUM_EPOCHS} | Treino Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            
            if val_acc > best_acc:
                best_acc = val_acc

        fold_results.append(best_acc)
        print(f"--> Melhor acurácia de validação para o Fold {fold+1}: {best_acc * 100:.2f}%")
    
    print(f'\n{"-"*20} Resultados Finais {"-"*20}')
    avg_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    print(f'Acurácia média final entre os folds: {avg_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)')

    print(f"\nTreinando modelo final com todos os dados por {config.NUM_EPOCHS} épocas...")
    dataset.transforms = data_transforms['train']
    full_data_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    final_model = get_model(pretrained=True).to(config.DEVICE)
    optimizer = optim.Adam(
        final_model.get_classifier().parameters(),
        lr=config.LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config.NUM_EPOCHS):
        _, train_acc = train_one_epoch(final_model, full_data_loader, criterion, optimizer, device=config.DEVICE)
        print(f'Época {epoch+1:02d}/{config.NUM_EPOCHS} | Treino Acurácia: {train_acc:.4f}')
        
    torch.save(final_model.state_dict(), config.MODEL_SAVE_DIR)
    print(f'Modelo final salvo em {config.MODEL_SAVE_DIR}')
    
if __name__ == "__main__":
    run_training()
    