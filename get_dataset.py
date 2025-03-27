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


def get_data_loaders(train_dir, test_dir, batch_size=32, num_workers=32, subset_size=0):
    """Возвращает DataLoader, загружая весь датасет, если subset_size == 0, или его часть, если subset_size > 0."""
    # Определяем устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Используется устройство: {device}")

    # Загружаем предобученные веса и трансформации для ResNet50
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    auto_transforms = weights.transforms()

    # Загружаем полный датасет
    train_data = datasets.ImageFolder(root=train_dir, transform=auto_transforms)
    test_data = datasets.ImageFolder(root=test_dir, transform=auto_transforms)

    # Проверяем значение subset_size
    if subset_size == 0:
        # Используем весь датасет
        train_subset = train_data
        test_subset = test_data
        logger.info(f"Используется весь тренировочный датасет: {len(train_data)} изображений")
        logger.info(f"Используется весь тестовый датасет: {len(test_data)} изображений")
    else:
        # Используем подмножество из первых subset_size изображений
        train_subset = Subset(train_data, range(min(len(train_data), subset_size)))
        test_subset = Subset(test_data, range(min(len(test_data), subset_size)))
        logger.info(f"Используется {len(train_subset)} тренировочных изображений и {len(test_subset)} тестовых.")

    # Создаем DataLoader'ы
    train_dataloader = DataLoader(dataset=train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(dataset=test_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Получаем имена классов
    class_names = train_data.classes
    return train_dataloader, test_dataloader, class_names


def walk_through_dir(dir_path):
    """Выводит содержимое директории."""
    logger.info(f"Проход по директории: {dir_path}")
    for dirpath, dirnames, filenames in os.walk(dir_path):
        logger.info(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


if __name__ == "__main__":
    # Пример использования
    logger.info("Начинаем работу.")
    download_dataset()

    train_dir = './data/real-vs-fake/train'
    test_dir = './data/real-vs-fake/test'

    logger.info("Получаем DataLoader для тренировочных и тестовых данных.")
    train_dataloader, test_dataloader, class_names = get_data_loaders(train_dir, test_dir)

    logger.info(f"Class names: {class_names}")
    walk_through_dir(train_dir)
