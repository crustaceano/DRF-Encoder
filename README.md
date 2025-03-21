# DevAIL

## Kryptonite ML Challenge  
**Цель проекта** — разработка модели распознавания лиц, устойчивой к дипфейкам, путем добавления их в обучающую выборку.  
**Задача** - обучить энкодер для распознавания лиц

## Начало работы:
### Окружение
Версия Python: `Python 3.10.12`.
  
Создание окружения:

```bash
VENV_DIR="VENV_PATH"
python3 -m virtualenv $VENV_DIR
source $VENV_DIR/bin/activate

pip install -r requirements.txt
```



## Данные

Перед началом работы необходимо загрузить данные и разместить их в папке `data`. 

- **Данные для обучения**: [Скачать по ссылке](https://storage.codenrock.com/companies/codenrock-13/contests/kryptonite-ml-challenge/train.zip)
- **Данные для теста**: [Скачать по ссылке](https://storage.codenrock.com/companies/codenrock-13/contests/kryptonite-ml-challenge/test_public.zip)

1. Создайте папку `data` в корневой директории проекта.
2. Загрузите данные по указанным ссылкам.
3. Поместите загруженные файлы в папку `data`.


## Описание данных

Данные - реальные и синтетически сгенерированные изображения лиц расположены в папке `data`.

```
data
├── test_public  # тестовые данные
│   ├── 00000000    # pair_id - ID сравниваемой пары сообщений
│   │   ├── 0.jpg
│   │   └── 1.jpg
│   ├── 00000001
│   │   ├── 0.jpg
│   │   └── 1.jpg
...
└── train   # данные для обучения модели
│   ├── meta.json 
│   ├── images 
│   │   ├── 00000000    # label - ID человека
│   │   │   ├── 0.jpg
│   │   │   ├── 1.jpg
│   │   │   ├── 2.jpg
│   │   │  ...
│   │   │   └── k_0.jpg
│   │   ├── 00000001
│   │   │   ├── 0.jpg
│   │   │   ├── 1.jpg
│   │   │   ├── 2.jpg
│   │   │  ...
│   │   │   └── k_1.jpg
...
```

meta.json имеет структуру face_index/i.jpg : is_deepfake. Ключ содержит информацию о индексе человека и номере изображения, значение равно 0 для реальных данных и равно 1 для синтетических данных. 

### Преобразование датасета

- Дипфейки выделяются в отдельные классы (каждому дипфейку присваивается уникальный лейбл).
- выбрасываем классы, в которых только один предствитель

**Код преобразования датасета**: [`transform_train_df.py`](transform_train_df.py).  
**Преобразованный датасет**: [`train_data.csv`](train_data.csv) и [`valid_data.csv`](valid_data.csv).

## Модель  
Мы протестировали различные модели, и наилучшие результаты показал **ViT-B/16 DINO** — визуальный трансформер с размером батча 16.  
**Скачать чекпойнт обученной модели, вы можете так:** `gdown --fuzzy "https://drive.google.com/file/d/1tWmIGZMIgNVzSmYzJvqdaZg9nP7iYUr9/view?usp=sharing" -O checkpoints/best_model.pth`


## Эксперименты  
Подробности о проведенных экспериментах доступны в файле [`experiments.md`](experiments.md).

## Обучение  
Обучение модели происходило в два этапа:  

### Pretrain 
   - Функция потерь: **Triplet Loss**  
   - Семплирование всех триплетов в батче  
   - Скорость обучения: `1e-4`  

### finetune
   - Семплирование **hard triplets** согласно статье [Mining Hard Triplets](https://arxiv.org/pdf/1703.07737)  
   - Реализация семплера: [`HardTripletsMiner`](https://github.com/OML-Team/open-metric-learning/blob/main/oml/miners/inbatch_hard_tri.py) из OML  
   - Скорость обучения: `1e-6`  
---

## Пример запуска обучения/инференса
### Обучение: 
`python train.py --device cuda --epochs 10 --batch_size 32 --model_path path/to/pretrained/model --miner hard`

**Аргументы командной строки**
- `--device`: Устройство для обучения (по умолчанию `cuda`, если доступно, иначе `cpu`).
- `--epochs`: Количество эпох для обучения (по умолчанию `3`).
- `--batch_size`: Размер батча для обучения (по умолчанию `32`).
- `--model_path`: Путь к предобученной модели (по умолчанию `None`).
- `--miner`: Тип минера для триплетной потери. Возможные значения: `hard`, `all`, `None` (по умолчанию `None`).

### Инференс
`python make_submission.py checkpoints/bst_vitb16_hard_miner.pth`

**Скорость инференса 210 - 230 кадров в секунду на V100**
- Разрешение входного изображения: 224х224
-Количество каналов: 3
-  Размер батча: 64






