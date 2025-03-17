import pandas as pd
import json
import os

def compress_labels(a:list):
  compressed = sorted(list(set(a)))
  mapping = {old: new for new, old in enumerate(compressed)}
  return [mapping[label] for label in a], compressed

META_PATH = 'data/train/meta.json'
TRAIN_PATH = 'train.csv'


if __name__ == '__main__':

    '''
    Выделяем дипфейки по людям в отдельные классы, 
    _____________________________________________
    после трансформации может, можем потерять некоторую информацию про лейблы, 
    в файле label_info.json сохраним словарь {label_id: (group_id, is_deepfake)}
    '''

    # meta_info - информация deepfake/real 1/0
    with open(META_PATH, 'r') as file:
        meta_info = json.load(file)
    df_train = pd.read_csv(TRAIN_PATH)
    labels, paths = df_train['label'].to_list(), df_train['path'].to_list()


    new_labels = []
    for label, path in zip(labels, paths):
      meta_path = os.path.join(*(path.split('/')[-2:]))
      if not meta_info[meta_path]:
        new_labels.append(2 * label)
      else:
        new_labels.append(2 * label + 1)

    new_labels, _ = compress_labels(new_labels)

    new_df_train = df_train.copy()
    new_df_train['label'] = new_labels

    df_filtered = new_df_train.groupby("label").filter(lambda x: len(x) >= 2)
    df_filtered['label'], _ =  compress_labels(df_filtered['label'].to_list())

    labels, paths = df_filtered['label'].to_list(), df_filtered['path'].to_list()
    labels_info = dict()
    for label, path in zip(labels, paths):
        meta_path = os.path.join(*(path.split('/')[-2:]))
        group_id = int(path.split('/')[-2])
        path.split('/')
        labels_info[label] = (group_id, meta_info[meta_path])

    # сохраняем информацию про лейблы
    with open('labels_info.json', 'w') as file:
        json.dump(labels_info, file)

    # сохраняем преобразовнные датасет
    df_filtered.to_csv('train_new.csv', index=False)