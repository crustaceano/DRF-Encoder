import random
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision.transforms as T
import os

from oml import datasets as d
from oml.inference import inference
from oml.losses import TripletLossWithMiner, ArcFaceLoss
from oml.metrics import calc_retrieval_metrics_rr
from oml.miners import AllTripletsMiner, HardTripletsMiner, NHardTripletsMiner
from oml.models import ViTExtractor, ResnetExtractor, ViTCLIPExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, AdaptiveThresholding
from oml.samplers import BalanceSampler, CategoryBalanceSampler

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc

from utils import DeepfakePairDataset, calculate_metrics

# Парсим аргументы командной строки
parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model checkpoint')
parser.add_argument('--miner', type=str, choices=['hard', 'all', None], default=None, help='Triplet loss miner type')

args = parser.parse_args()

# Используем аргументы
device = args.device
epochs = args.epochs
batch_size = args.batch_size
model_path = args.model_path
miner = args.miner


CHECKPOITNS_DIR = 'checkpoints'
train_df = pd.read_csv('train_data.csv')
valid_df = pd.read_csv('valid_data.csv')


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def enable_dropout(model, new_dropout_rate=0.1):
    # Измените значение p для всех слоев Dropout
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = new_dropout_rate
            # print(f"Изменен {name}: p={module.p}")

if __name__ == "__main__":
    fix_seed(seed=0)

    if not model_path:
        model = ViTExtractor.from_pretrained("vitb16_dino").to(device)
    else:
        model = ViTExtractor(model_path, "vitb16", True).to(device)
    
    
    transform, _ = get_transforms_for_pretrained("vitb16_dino")
    
    df_train = train_df
    train = d.ImageLabeledDataset(df_train, transform=transform)
    valid_dataset = DeepfakePairDataset(valid_df, transform=transform)
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True, 
    )
    
    optimizer = Adam(model.parameters(), lr=1e-4)

    if miner=='hard':
        criterion = TripletLossWithMiner(0.1, HardTripletsMiner(), need_logs=True)
    else:
        criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
    # criterion = ArcFaceLoss(in_features, num_classes).to(device)

    sampler = BalanceSampler(train.get_labels(), n_labels=16, n_instances=4)
    # sampler = CategoryBalanceSampler(
    #     train.get_labels(), label2category, n_categories=2, n_labels=32, n_instances=4, resample_labels=False, weight_categories=True
    # ) 
    def train_epoch(epoch, checkpoint_name=None):
        model.train()
        avg_loss = 0.0
        pbar = tqdm(DataLoader(train, batch_sampler=sampler))
        pbar.set_description(f"epoch: {epoch}/{epochs}")
            
        for batch in pbar:
            embeddings = model(batch["input_tensors"].to(device))
            loss = criterion(embeddings, batch["labels"].to(device))
                
            avg_loss+=loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix(criterion.last_logs)

        avg_loss /= len(pbar)
        print(f'average loss: {avg_loss}')
        # print(f'{calc_train_metrics(model, val_loader)}')
        if checkpoint_name:
            torch.save(model.state_dict(), os.path.join(CHECKPOITNS_DIR, f"{checkpoint_name}.pth"))
        return {'avg_loss': avg_loss}

    def validation_epoch(epoch):
        model.eval()
        # val_loss = 0.0
        sim_pred = []
        y_true = []
        with torch.no_grad():
            for x1, x2, y in tqdm(val_loader):  # <-- И здесь
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                embs1 = model(x1)
                embs2 = model(x2)
                sim_pred += F.cosine_similarity(embs1, embs2).cpu().numpy().tolist()
                y_true += y.cpu().numpy().tolist()

        metrics = calculate_metrics(np.array(y_true), np.array(sim_pred))
        print(f"Epoch {epoch+1} Validation metrics: {metrics}")
        return metrics

    
    train_losses = []
    val_metrics_list = []

    for epoch in range(epochs):
        loss = train_epoch(epoch, checkpoint_name=f"checkpoint_epoch_{epoch+1}")
        train_losses.append(loss)
        metrics = validation_epoch(epoch)
        val_metrics_list.append(metrics)
    
    torch.save(model.state_dict(), "model_final.pth")
    
    # Построение графиков обучения и валидации
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(12, 5))
    
    # График тренировочной потери
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # График EER на валидации
    plt.subplot(1, 2, 2)
    EER = [m['EER'] for m in val_metrics_list]
    plt.plot(epochs_range, EER, marker='o', label='Validation EER', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('EER')
    plt.title('Validation EER')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
