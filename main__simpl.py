import os
import dotenv
import logging
import numpy as np
from PIL import Image
import io
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import joblib

# Мы убрали импорт из train_model, чтобы избежать ошибки "No module named..."
# Теперь архитектура прописана прямо здесь.

from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен бота
dotenv.load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# ==========================================
# ВСТРОЕННАЯ АРХИТЕКТУРА И ФУНКЦИИ
# ==========================================

class SimpleCNN(nn.Module):
    """
    Архитектура нейросети.
    Дублируется здесь, чтобы бот мог работать автономно без файла train_model.py
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2), # 2 класса: Real (0) и AI (1)
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def extract_features(image_array):
    """Извлечение признаков (яркость, контраст, гистограммы)"""
    try:
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

        features = {}
        # Яркость и контраст
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        features['brightness'] = np.mean(gray)
        features['contrast'] = np.std(gray)

        # Градиенты (резкость)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        features['gradient_mean'] = np.mean(gradient)

        # Гистограмма
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-6)

        # Собираем вектор
        feature_vector = [features['brightness'], features['contrast'], features['gradient_mean']]
        feature_vector.extend(hist)

        return np.array(feature_vector)
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return np.zeros(19)

# ==========================================
# ЛОГИКА БОТА
# ==========================================

class AIDetectorBot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")

        # Создаем папки для сохранения пользовательских фото
        os.makedirs('user_images', exist_ok=True)

        # Загрузка моделей
        self.load_models()

        # Трансформации (должны совпадать с теми, что были при обучении)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.stats = {'total': 0, 'ai': 0, 'real': 0}

    def load_models(self):
        """Загрузка обученных весов"""
        print("🔄 Загрузка моделей...")
        try:
            # 1. CNN
            self.cnn_model = SimpleCNN().to(self.device)
            if os.path.exists('models/cnn_model.pth'):
                self.cnn_model.load_state_dict(torch.load('models/cnn_model.pth', map_location=self.device))
                self.cnn_model.eval() # Переключаем в режим предсказания
                print("✅ CNN модель загружена")
            else:
                print("❌ Файл models/cnn_model.pth не найден! Сначала запустите train_model.py")

            # 2. Random Forest & Scaler
            if os.path.exists('models/classifier.joblib') and os.path.exists('models/scaler.joblib'):
                self.feature_classifier = joblib.load('models/classifier.joblib')
                self.scaler = joblib.load('models/scaler.joblib')
                print("✅ ML классификаторы загружены")
            else:
                print("❌ Файлы joblib не найдены! Сначала запустите train_model.py")

        except Exception as e:
            print(f"🔥 Критическая ошибка загрузки: {e}")

    def analyze_image(self, image_rgb, user_id):
        """Полный цикл анализа"""
        results = {}

        try:
            # --- 1. Анализ CNN ---
            pil_image = Image.fromarray(image_rgb)
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.cnn_model(img_tensor)
                # outputs = [prob_real, prob_ai]
                probs = outputs.cpu().numpy()[0]

            results['cnn_real'] = float(probs[0])
            results['cnn_ai'] = float(probs[1])

            # --- 2. Анализ признаков (ML) ---
            features = extract_features(image_rgb)
            # Внимание: feature_classifier ожидает 2D массив
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            ml_probs = self.feature_classifier.predict_proba(features_scaled)[0]

            results['feature_real'] = float(ml_probs[0])
            results['feature_ai'] = float(ml_probs[1])

            # --- 3. Артефакты (OpenCV) ---
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            results['sharpness'] = laplacian

            # --- Итоговый вердикт ---
            # Усредняем вероятности (можно добавить веса)
            final_ai_prob = (results['cnn_ai'] * 0.6) + (results['feature_ai'] * 0.4)
            results['combined_ai'] = final_ai_prob

            if final_ai_prob > 0.65:
                results['verdict'] = "🤖 AI GENERATED"
                self.stats['ai'] += 1
            elif final_ai_prob < 0.35:
                results['verdict'] = "📸 REAL PHOTO"
                self.stats['real'] += 1
            else:
                results['verdict'] = "❓ UNCERTAIN"

            self.stats['total'] += 1

            return results

        except Exception as e:
            logger.error(f"Analysis Failed: {e}")
            return None

    def create_result_vis(self, image_rgb, results):
        """Рисуем красивый ответ на картинке"""
        vis_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        color = (0, 0, 255) if results['combined_ai'] > 0.5 else (0, 255, 0)
        text = f"{results['verdict']} ({results['combined_ai']*100:.1f}%)"

        # Подложка под текст
        cv2.rectangle(vis_img, (0, 0), (vis_img.shape[1], 60), (0,0,0), -1)
        cv2.putText(vis_img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                   1, color, 2)

        return vis_img

# Глобальный экземпляр
detector = AIDetectorBot()

# --- Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Привет! Я готов. Сначала убедись, что администратор запустил обучение моделей.\nОтправь мне фото!")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("⏳ Думаю...")
    try:
        photo = await update.message.photo[-1].get_file()
        photo_bytes = await photo.download_as_bytearray()
        nparr = np.frombuffer(photo_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = detector.analyze_image(img_rgb, update.effective_user.id)

        if res:
            vis_img = detector.create_result_vis(img_rgb, res)

            # Отправка
            _, buffer = cv2.imencode('.jpg', vis_img)
            await update.message.reply_photo(
                photo=io.BytesIO(buffer),
                caption=f"🧠 **AI Probability:** {res['combined_ai']*100:.1f}%\n"
                        f"📊 **CNN:** {res['cnn_ai']:.2f} | **ML:** {res['feature_ai']:.2f}",
                parse_mode=ParseMode.MARKDOWN
            )
            await status_msg.delete()
        else:
            await status_msg.edit_text("❌ Ошибка анализа.")

    except Exception as e:
        logger.error(e)
        await status_msg.edit_text("Ошибка обработки.")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("🚀 Бот запущен!")
    app.run_polling()

if __name__ == "__main__":
    main()
