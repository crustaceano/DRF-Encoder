import pandas as pd
import random
from itertools import product, combinations
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc


def generate_cat1_pairs(df):
    '''Пары с одинаковым id и оба is_deepfake=0 (distance=0)'''
    filtered = df[df['is_deepfake'] == 0]
    groups = filtered.groupby('label').filter(lambda x: len(x) >= 2)
    pairs = []
    for _, group in groups.groupby('label'):
        paths = group['path'].tolist()
        pairs.extend(combinations(paths, 2))
    return pairs

def generate_cat2_pairs(df):
    '''Пары с одинаковым id и разными is_deepfake (0 и 1) (distance=1)'''
    pairs = []
    for id_num, group in df.groupby('label'):
        real = group[group['is_deepfake'] == 0]['path'].tolist()
        fake = group[group['is_deepfake'] == 1]['path'].tolist()
        if real and fake:
            pairs.extend(list(product(real, fake)))
    return pairs

def generate_cat3_pairs(df, n=None):
    '''Пары с разными id и не оба is_deepfake=1 (distance=1)'''
    candidates = df[df['is_deepfake'] == 0]
    pairs = []
    ids = candidates['label'].unique()

    while len(pairs) < n:
        try:
            id1, id2 = random.sample(list(ids), 2)
        except:
            break

        pair = candidates[candidates['label'].isin([id1, id2])]
        if len(pair) >= 2:
            selected = pair.sample(2)
            p1, p2 = selected.iloc[0]['path'], selected.iloc[1]['path']
            pairs.append((p1, p2))
    if n is None or n > len(pairs):
      n = len(pairs)
    return pairs[:n]

import json

def get_sampling_result(df, k = 10000):
    '''
    transform dataframe

    output:  DataFrame (path1, path2, label)
    '''
    
    with open('data/train/meta.json', 'r') as file:
        meta = json.load(file)
    # df['path'] = df['path'].apply(lambda x: "/".join(x.split('/')[-2:]))
    df['path'] = df['path'].apply(lambda x: "/".join(x.split('/')[-2:]))
    df['is_deepfake'] = df['path'].apply(lambda x: meta[x])
    df = df.drop(columns=['split', 'is_query', 'is_gallery'])
    
    cat1 = generate_cat1_pairs(df)
    cat2 = generate_cat2_pairs(df)
    cat3 = generate_cat3_pairs(df, len(cat1)*2)
    
    
    final_pairs = (
        [[p1, p2, 0] for p1, p2 in cat1] +
        [[p1, p2, 1] for p1, p2 in cat2] +
        [[p1, p2, 1] for p1, p2 in cat3]
    )
    
    # random.shuffle(final_pairs)
    if k:
        final_pairs = random.sample(final_pairs, k)
        
    random.shuffle(final_pairs)
    return pd.DataFrame(final_pairs, columns=['path1', 'path2', 'distance'])

class DeepfakePairDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Загрузка изображений train/images/000000/4.jpg
        img1 = Image.open('data/train/images/' + row['path1']).convert('RGB')
        img2 = Image.open('data/train/images/' + row['path2']).convert('RGB')

        # Применение трансформаций
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(row['distance'], dtype=torch.float32)

        return img1, img2, label


def calculate_metrics(y_true, y_pred):
    y_pred = (y_pred + 1) / 2
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    return {
        'roc_auc': roc_auc_score(y_true, y_pred),
        'pr_auc': average_precision_score(y_true, y_pred),
        # 'accuracy': (y_pred.round() == y_true).mean(),
        'EER': eer_threshold
    }
