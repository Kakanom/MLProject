import torch
import torchvision
from torchvision import datasets, transforms
import opendatasets as od
from torch.utils.data import DataLoader
from pathlib import Path
import os
import logging

from torch.utils.data import Subset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def download_dataset():
    data_dir = './data'

    if os.path.exists(data_dir) and os.listdir(data_dir):
        logger.info(f"Папка {data_dir} уже существует и не пуста. Пропускаем загрузку.")
    else:
        logger.info(f"Папка {data_dir} не существует или пуста. Загружаем датасет.")
        dataset_url = 'https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces'
        try:
            od.download(dataset_url, data_dir=data_dir)
            logger.info(f"Датасет успешно загружен в {data_dir}.")
        except Exception as e:
            logger.error(f"Ошибка при скачивании датасета: {e}")
            raise


def get_data_loaders(train_dir, val_dir, test_dir, batch_size=32, num_workers=32, subset_size=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Используется устройство: {device}")

    # Трансформации ResNet50
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    auto_transforms = weights.transforms()

    # Загрузка ImageFolder'ов
    train_data = datasets.ImageFolder(root=train_dir, transform=auto_transforms)
    val_data = datasets.ImageFolder(root=val_dir, transform=auto_transforms)
    test_data = datasets.ImageFolder(root=test_dir, transform=auto_transforms)

    if subset_size == 0:
        train_subset = train_data
        val_subset = val_data
        test_subset = test_data
        logger.info(f"Тренировочных: {len(train_data)}, валидационных: {len(val_data)}, тестовых: {len(test_data)}")
    else:
        train_subset = Subset(train_data, range(min(len(train_data), subset_size)))
        val_subset = Subset(val_data, range(min(len(val_data), subset_size)))
        test_subset = Subset(test_data, range(min(len(test_data), subset_size)))
        logger.info(f"Используем подмножество: train={len(train_subset)}, val={len(val_subset)}, test={len(test_subset)}")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_data.classes
    return train_loader, val_loader, test_loader, class_names



def walk_through_dir(dir_path):
    """Выводит содержимое директории."""
    logger.info(f"Проход по директории: {dir_path}")
    for dirpath, dirnames, filenames in os.walk(dir_path):
        logger.info(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


if __name__ == "__main__":
    download_dataset()

    train_dir = './data/real-vs-fake/train'
    val_dir = './data/real-vs-fake/valid'
    test_dir = './data/real-vs-fake/test'

    logger.info("Получаем DataLoader для тренировочных и тестовых данных.")
    
    train_loader, val_loader, test_loader, class_names = get_data_loaders(train_dir, val_dir, test_dir)

    logger.info(f"Class names: {class_names}")
    walk_through_dir(train_dir)
