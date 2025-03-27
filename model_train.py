import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchinfo import summary
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def create_model(num_classes=2, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Создаёт модель ResNet50 с кастомным классификатором."""
    logger.info(f"Creating model with {num_classes} classes on device {device}.")
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).to(device)

    # Заморозка слоёв
    for param in model.parameters():
        param.requires_grad = False

    # Новый классификатор
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.Dropout(),
        nn.Linear(500, num_classes)
    ).to(device)

    logger.info("Model created successfully.")
    return model


def train_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    logger.info(f"Train step: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
    return train_loss, train_acc


def test_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    logger.info(f"Test step: Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}")
    return test_loss, test_acc


def train(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer, loss_fn: nn.Module, epochs: int, device):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in tqdm(range(epochs)):
        logger.info(f"Epoch {epoch + 1}/{epochs} starting...")

        # Train step
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)

        # Test step
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        logger.info(f"Epoch {epoch + 1}: train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, "
                    f"test_loss = {test_loss:.4f}, test_acc = {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


def save_model(model, path="models/RealityCheck.pth"):
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


if __name__ == "__main__":
    from get_dataset import get_data_loaders

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dir = './data/real-vs-fake/train'
    test_dir = './data/real-vs-fake/test'

    # Загрузка данных
    logger.info("Loading dataset...")
    train_dataloader, test_dataloader, class_names = get_data_loaders(train_dir, test_dir)
    logger.info(f"Dataset loaded with {len(class_names)} classes.")

    # Создание модели
    model = create_model(num_classes=len(class_names), device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Обучение
    logger.info("Training started...")
    results = train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs=5, device=device)

    # Сохранение модели
    save_model(model)

    # Вывод структуры модели
    logger.info("Model summary:")
    summary(model, input_size=(32, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"])
