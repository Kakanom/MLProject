import os
import torch
import torchvision
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from model_train import create_model
from torchvision import transforms
from config import *
from telegram.ext import Application, CommandHandler, MessageHandler, filters, \
    CallbackContext
from telegram import Update, ReplyKeyboardMarkup
from PIL import Image
import logging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Загрузка модели
def load_model(model_path="models/ResNet50_30epoch.pth", device=DEVICE):
    logger.info(f"Загрузка модели с пути: {model_path}")
    try:
        model = create_model(num_classes=2, device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info("Модель успешно загружена и переведена в режим оценки (eval)")
        return model
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        raise


# Веса
def get_transforms(weight_path="models/ResNet50.pth"):
    logger.info("Получение трансформаций для изображения")
    try:
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        return weights.transforms()
    except Exception as e:
        logger.error(f"Ошибка при получении трансформаций: {e}")
        raise


def predict_image(image_path, model, transform, class_names=None, device=DEVICE):
    if class_names is None:
        class_names = ['Fake', 'Real']
    logger.info(f"Предсказание для изображения: {image_path}")
    try:
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            pred_logits = model(img)
            logger.info(f"Логиты модели: {pred_logits.cpu().numpy()}")  # Выводим сырые логиты

            pred_probs = torch.softmax(pred_logits, dim=1)
            logger.info(f"Вероятности классов (после softmax): {pred_probs.cpu().numpy()}")  # Выводим вероятности

            pred_label = torch.argmax(pred_probs, dim=1).item()
            prob = pred_probs.max().item() * 100

        logger.info(f"Предсказано: {class_names[pred_label]} с вероятностью {prob:.2f}%")
        return class_names[pred_label], prob

    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise


# --- Команда /start ---
async def start(update: Update, context: CallbackContext):
    logger.info("Обработан запрос /start")
    keyboard = [["📷 Отправить фото"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text(
        "Привет! Я бот IsRealBot. Отправь мне фото, и я скажу, настоящее оно или сгенерировано нейросетью!",
        reply_markup=reply_markup
    )


# --- Обработчик фото ---
async def handle_photo(update: Update, context: CallbackContext):
    logger.info("Получено фото от пользователя")
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_path = f"received_{photo.file_id}.jpg"
    await file.download_to_drive(file_path)
    logger.info(f"Фото загружено и сохранено как {file_path}")

    model = load_model()
    transform = get_transforms()
    pred_label, prob = predict_image(file_path, model, transform)

    if prob > 60:
        response = f"Я уверен на {prob:.2f}%, что это {pred_label.lower()} фото.\n"
        if pred_label == 'Fake':
            response += "Скорее всего, оно сгенерировано нейросетью!"
        else:
            response += "Это выглядит как настоящее фото."
    else:
        response = f"Вероятность, что фото {pred_label.lower()}: {prob:.2f}%.\n" \
                   f"Я не совсем уверен, но склоняюсь к тому, что это {pred_label.lower()}."

    await update.message.reply_text(response)
    logger.info(f"Ответ отправлен пользователю: {response}")

    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Временный файл {file_path} удален")


# --- Главная функция ---
def main():
    logger.info("Запуск бота...")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    logger.info("Бот запущен и готов к работе.")
    app.run_polling()


if __name__ == "__main__":
    main()
