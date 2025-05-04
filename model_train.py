import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchinfo import summary
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from plot_metrics import save_plot_metrics
from get_dataset import get_data_loaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

name = ""


def create_model(num_classes=2, device="cuda" if torch.cuda.is_available() else "cpu"):
    logger.info(f"Creating model with {num_classes} classes on device {device}.")
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).to(device)

    for param in model.parameters():
        param.requires_grad = False

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
    all_preds, all_labels = [], []
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

        all_preds.extend(y_pred_class.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    train_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    train_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    logger.info(
        f"Train step: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}, Precision = {train_precision:.4f}, Recall = {train_recall:.4f}, F1 = {train_f1:.4f}")
    return train_loss, train_acc, train_precision, train_recall, train_f1


def test_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device):
    model.eval()
    test_loss, test_acc = 0, 0
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

            all_preds.extend(test_pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    test_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    logger.info(
        f"Test step: Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}, Precision = {test_precision:.4f}, Recall = {test_recall:.4f}, F1 = {test_f1:.4f}")
    return test_loss, test_acc, test_precision, test_recall, test_f1


def train(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer, loss_fn: nn.Module, epochs: int, device):
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_precision": [], "val_recall": [],
               "val_f1": [], "train_precision": [], "train_recall": [],
               "train_f1": []}
    for epoch in tqdm(range(epochs)):
        logger.info(f"Epoch {epoch + 1}/{epochs} starting...")

        train_loss, train_acc, train_precision, train_recall, train_f1 = train_step(model, train_dataloader, loss_fn,
                                                                                    optimizer, device)
        val_loss, val_acc, val_precision, val_recall, val_f1 = test_step(model, val_dataloader, loss_fn, device)

        logger.info(f"Epoch {epoch + 1}: train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, "
                    f"val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        results["train_precision"].append(train_precision)
        results["train_recall"].append(train_recall)
        results["train_f1"].append(train_f1)

        results["val_precision"].append(val_precision)
        results["val_recall"].append(val_recall)
        results["val_f1"].append(val_f1)
    return results


def save_model(model, path=f"models/"):
    path += f"{name}.pth"
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


def plot_and_save_metrics(results, save_path=f"plots/"):
    epochs = range(1, len(results["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(3, 2, 1)
    plt.plot(epochs, results["train_acc"], label="Train Accuracy")
    plt.plot(epochs, results["val_acc"], label="Val Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(epochs, results["train_precision"], label="Train Precision")
    plt.plot(epochs, results["val_precision"], label="Val Precision")
    plt.title("Precision")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(epochs, results["train_recall"], label="Train recall")
    plt.plot(epochs, results["val_recall"], label="Val recall")
    plt.title("Recall")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(epochs, results["train_f1"], label="Train F1")
    plt.plot(epochs, results["val_f1"], label="Val F1")
    plt.title("F1")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(epochs, results["train_loss"], label="Train Loss")
    plt.plot(epochs, results["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path + "METRICS" + f"{name}.png")
    logger.info(f"METRICS plot saved to {save_path}METRICS{name}.png")
    plt.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    name = input("Enter name of model: ")

    if name == "":
        name = "test"

    train_dir = './data/real-vs-fake/train'
    val_dir = './data/real-vs-fake/valid'
    test_dir = './data/real-vs-fake/test'

    logger.info("Loading dataset...")
    train_dataloader, val_dataloader, test_dataloader, class_names = get_data_loaders(train_dir, val_dir, test_dir)
    logger.info(f"Dataset loaded with {len(class_names)} classes.")

    model = create_model(num_classes=len(class_names), device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logger.info("Training started...")
    results = train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs=10, device=device)

    plot_and_save_metrics(results)
    save_model(model)

    logger.info("Evaluating on test dataset...")
    final_test_loss, final_test_acc, precision, recall, f1 = test_step(model, test_dataloader, loss_fn, device)
    logger.info(
        f"Final Test Loss: {final_test_loss:.4f}, Final Test Accuracy: {final_test_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    logger.info("Model summary:")
    summary(model, input_size=(32, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"])
