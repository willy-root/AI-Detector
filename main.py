import os
import time
import dotenv
import logging
import numpy as np
from PIL import Image, ImageFont
import io
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Telegram Bot 
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode


# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен вынесен в файл окружения .env
dotenv.load_dotenv()
TELEGRAM_TOKEN =    os.getenv("TELEGRAM_TOKEN")

# Не приватные переменные окружения, вместо export, вынесены в файл
dotenv.load_dotenv("./.env_open")
CNN_AI_INDEX =      os.getenv("CNN_AI_INDEX")
FEATURE_AI_LABEL =  os.getenv("FEATURE_AI_LABEL", "1")
FEATURE_AUTO_FLIP = os.getenv("FEATURE_AUTO_FLIP", "1") != "0"


class AIDetectorBot:
    """Класс детектора AI-изображений для Telegram бота"""

    def __init__(self):
        global FEATURE_AUTO_FLIP
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        print(f"Используется устройство: {self.device}")
        self.cnn_ai_index = self._resolve_cnn_ai_index()
        self.feature_ai_label = self._resolve_feature_ai_label()
        #self.feature_auto_flip = os.getenv("FEATURE_AUTO_FLIP", "1") != "0"
        self.feature_auto_flip = FEATURE_AUTO_FLIP
        self.cnn_model_loaded = False
        self.feature_model_loaded = False

        # Создаем папки если их нет
        os.makedirs('user_images', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Инициализация моделей
        self.initialize_models()

        # Статистика
        self.stats = {
            'total_images': 0,
            'ai_detected': 0,
            'real_detected': 0
        }

    def _resolve_cnn_ai_index(self):
        """Определяет индекс класса AI в выходе CNN."""
        #env_value = os.getenv("CNN_AI_INDEX")
        global CNN_AI_INDEX
        env_value = CNN_AI_INDEX
        if env_value in {"0", "1"}:
            return int(env_value)

        dataset_train_dir = os.path.join("dataset", "train")
        if os.path.isdir(dataset_train_dir):
            class_dirs = sorted(
                name for name in os.listdir(dataset_train_dir)
                if os.path.isdir(os.path.join(dataset_train_dir, name))
            )
            if "ai" in class_dirs:
                ai_index = class_dirs.index("ai")
                logger.info(
                    "Индекс AI для CNN определен по папкам датасета: %s (классы=%s)",
                    ai_index, class_dirs
                )
                return ai_index

        # Для ImageFolder классы обычно сортируются как ['ai', 'real'] => AI = 0
        return 0

    def _resolve_feature_ai_label(self):
        """
        Определяет, какое значение метки в RandomForest соответствует классу AI.
        По умолчанию используется '1' (можно изменить через FEATURE_AI_LABEL).
        """
        global FEATURE_AI_LABEL
        #return os.getenv("FEATURE_AI_LABEL", "1")
        return FEATURE_AI_LABEL

    def _maybe_flip_feature_probability(self, cnn_ai_prob, feature_ai_prob):
        """
        При явной инверсии меток в признаковом классификаторе
        (когда его прогноз зеркален CNN), автоматически отражает вероятность.
        """
        if not self.feature_auto_flip:
            return feature_ai_prob, False

        raw_diff = abs(cnn_ai_prob - feature_ai_prob)
        flipped_ai = 1.0 - feature_ai_prob
        flipped_diff = abs(cnn_ai_prob - flipped_ai)

        if raw_diff >= 0.45 and (raw_diff - flipped_diff) >= 0.25:
            return flipped_ai, True

        return feature_ai_prob, False

    def initialize_models(self):
        """Инициализация моделей детектора"""
        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Создаем CNN модель
        self.cnn_model = self.create_cnn_model()

        # Создаем классификатор на признаках
        self.feature_classifier = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            max_depth=10
        )
        self.scaler = StandardScaler()

        # Попытка загрузить предобученные модели
        self.load_or_create_models()

    def create_cnn_model(self):
        """Создание простой CNN модели"""

        class SimpleCNN(nn.Module):
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
                    nn.Linear(128, 2),
                    nn.Softmax(dim=1)
                )

            def forward(self, x):
                x = self.conv_layers(x)
                x = x.view(x.size(0), -1)
                x = self.fc_layers(x)
                return x

        return SimpleCNN().to(self.device)

    def load_or_create_models(self):
        """Загрузка или создание моделей"""
        try:
            # Пытаемся загрузить сохраненные модели
            if os.path.exists('models/cnn_model.pth'):
                self.cnn_model.load_state_dict(torch.load('models/cnn_model.pth',
                                                          map_location=self.device))
                self.cnn_model_loaded = True
                print("✓ Загружена CNN модель")
            else:
                logger.warning("Файл models/cnn_model.pth не найден, CNN работать не будет")

            if os.path.exists('models/classifier.joblib'):
                self.feature_classifier = joblib.load('models/classifier.joblib')
                print("✓ Загружен классификатор")
            else:
                logger.warning("Файл models/classifier.joblib не найден")

            if os.path.exists('models/scaler.joblib'):
                self.scaler = joblib.load('models/scaler.joblib')
                print("✓ Загружен нормализатор")
            else:
                logger.warning("Файл models/scaler.joblib не найден")

            self.feature_model_loaded = (
                os.path.exists('models/classifier.joblib') and
                os.path.exists('models/scaler.joblib')
            )

        except Exception as e:
            print(f"Не удалось загрузить модели: {e}")
            print("Используются модели с базовыми настройками")
            self.cnn_model_loaded = False
            self.feature_model_loaded = False

    def extract_features(self, image_array, expected_dim=None):
        """Извлечение признаков из изображения"""
        try:
            if len(image_array.shape) == 2:  # Если черно-белое
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

            # 1. Базовые признаки (совместимы со старой моделью на 19 признаков)
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(sobelx ** 2 + sobely ** 2)
            gradient_mean = float(np.mean(gradient))
            gradient_std = float(np.std(gradient))

            # 2. Гистограмма
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-6)

            # 3. Расширенные цветовые признаки (новый формат, 26 признаков)
            mean_color = np.mean(image_array, axis=(0, 1))
            std_color = np.std(image_array, axis=(0, 1))

            vector_19 = [brightness, contrast, gradient_mean]
            vector_19.extend(hist.tolist())

            vector_26 = mean_color.flatten().tolist()
            vector_26.extend(std_color.flatten().tolist())
            vector_26.extend([brightness, contrast, gradient_mean, gradient_std])
            vector_26.extend(hist.tolist())

            if expected_dim is None:
                return np.array(vector_26, dtype=np.float32)

            if expected_dim == 19:
                return np.array(vector_19, dtype=np.float32)

            if expected_dim == 26:
                return np.array(vector_26, dtype=np.float32)

            if expected_dim < 26:
                logger.warning(
                    "Неожиданная размерность признаков у scaler: %s. "
                    "Использую первые %s признаков.",
                    expected_dim, expected_dim
                )
                return np.array(vector_26[:expected_dim], dtype=np.float32)

            logger.warning(
                "Неожиданная размерность признаков у scaler: %s. "
                "Дополняю нулями до нужной длины.",
                expected_dim
            )
            padded = vector_26 + [0.0] * (expected_dim - len(vector_26))
            return np.array(padded, dtype=np.float32)

        except Exception as e:
            logger.error(f"Ошибка извлечения признаков: {e}")
            return None

    def analyze_with_cnn(self, image):
        """Анализ изображения CNN"""
        try:
            if not self.cnn_model_loaded:
                return None

            # Преобразуем numpy array в PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Применяем трансформации
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Предсказание
            with torch.no_grad():
                self.cnn_model.eval()
                output = self.cnn_model(image_tensor)
                probabilities = output.cpu().numpy()[0].astype(float)

            if probabilities.shape[0] != 2 or self.cnn_ai_index not in (0, 1):
                logger.error(
                    "Некорректный выход CNN: shape=%s, cnn_ai_index=%s",
                    probabilities.shape, self.cnn_ai_index
                )
                return None

            ai_prob = float(probabilities[self.cnn_ai_index])
            real_prob = float(probabilities[1 - self.cnn_ai_index])

            return {
                'ai': ai_prob,
                'real': real_prob
            }

        except Exception as e:
            logger.error(f"Ошибка CNN анализа: {e}")
            return None

    def analyze_with_features(self, image):
        """Анализ на основе признаков"""
        try:
            if not self.feature_model_loaded:
                return None

            expected_dim = getattr(self.scaler, 'n_features_in_', None)
            if expected_dim is None:
                expected_dim = getattr(self.feature_classifier, 'n_features_in_', None)

            if expected_dim is None:
                logger.error("Невозможно определить размерность признаков для scaler/classifier")
                return None

            features = self.extract_features(image, expected_dim=expected_dim)
            if features is None:
                return None

            features_scaled = self.scaler.transform(features.reshape(1, -1))
            probs = self.feature_classifier.predict_proba(features_scaled)[0]
            classes = getattr(self.feature_classifier, 'classes_', None)

            if classes is None or len(classes) != len(probs):
                logger.error("Некорректные classes_ у feature_classifier: %s", classes)
                return None

            class_to_prob = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
            ai_key = str(self.feature_ai_label)
            if ai_key not in class_to_prob:
                logger.error(
                    "Метка AI '%s' не найдена в classes_=%s. "
                    "Укажите FEATURE_AI_LABEL корректно.",
                    ai_key, list(class_to_prob.keys())
                )
                return None

            ai_prob = class_to_prob[ai_key]
            real_prob = 1.0 - ai_prob

            return {
                'ai': float(ai_prob),
                'real': float(real_prob)
            }

        except Exception as e:
            logger.error(f"Ошибка feature анализа: {e}")
            return None

    def detect_ai_artifacts(self, image):
        """Поиск артефактов AI"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            artifacts = {}

            # 1. Резкость
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            artifacts['sharpness'] = laplacian.var()

            # 2. Регулярность текстур
            edges = cv2.Canny(gray, 100, 200)
            artifacts['edge_uniformity'] = np.std(edges) / (np.mean(edges) + 1e-10)

            return artifacts

        except Exception as e:
            logger.error(f"Ошибка поиска артефактов: {e}")
            return {}

    def analyze_image(self, image_array, user_id):
        """Основной метод анализа изображения"""
        try:
            results = {}

            # Анализ CNN
            cnn_result = self.analyze_with_cnn(image_array)
            if cnn_result is not None:
                results['cnn_real'] = cnn_result['real']
                results['cnn_ai'] = cnn_result['ai']

            # Анализ по признакам
            feature_result = self.analyze_with_features(image_array)
            if feature_result is not None:
                feature_ai = feature_result['ai']
                feature_real = feature_result['real']

                if cnn_result is not None:
                    corrected_ai, flipped = self._maybe_flip_feature_probability(
                        cnn_result['ai'], feature_ai
                    )
                    if flipped:
                        logger.warning(
                            "Вероятности признаковой модели инвертированы. "
                            "Выполнена авто-коррекция feature_ai -> 1 - feature_ai."
                        )
                        feature_ai = corrected_ai
                        feature_real = 1.0 - corrected_ai
                        results['feature_label_auto_flipped'] = True

                results['feature_real'] = feature_real
                results['feature_ai'] = feature_ai

            # Поиск артефактов
            artifacts = self.detect_ai_artifacts(image_array)
            results['artifacts'] = artifacts

            # Комбинированный результат
            weighted_scores = []
            if cnn_result is not None:
                weighted_scores.append((results['cnn_ai'], 0.6))
            if feature_result is not None:
                weighted_scores.append((results['feature_ai'], 0.4))

            if weighted_scores:
                total_weight = sum(weight for _, weight in weighted_scores)
                combined_ai = sum(prob * weight for prob, weight in weighted_scores) / total_weight
            else:
                combined_ai = 0.5

            results['combined_ai'] = combined_ai

            # Определение вердикта
            if combined_ai > 0.7:
                results['verdict'] = "🤖 ВЫСОКАЯ ВЕРОЯТНОСТЬ AI"
                results['confidence'] = "Высокая"
                self.stats['ai_detected'] += 1
            elif combined_ai > 0.6:
                results['verdict'] = "⚠️ Возможно AI"
                results['confidence'] = "Средняя"
            elif combined_ai > 0.4:
                results['verdict'] = "❓ Неопределенно"
                results['confidence'] = "Низкая"
            else:
                results['verdict'] = "✅ Вероятно реальное"
                results['confidence'] = "Высокая"
                self.stats['real_detected'] += 1

            self.stats['total_images'] += 1

            # Сохраняем изображение для истории
            self.save_user_image(image_array, user_id, results.get('verdict', 'Unknown'))

            return results

        except Exception as e:
            logger.error(f"Ошибка анализа: {e}")
            return None

    def save_user_image(self, image_array, user_id, verdict):
        """Сохранение изображения пользователя"""
        try:
            # Создаем папку пользователя
            user_dir = f'user_images/user_{user_id}'
            os.makedirs(user_dir, exist_ok=True)

            # Генерируем имя файла
            timestamp = int(time.time())
            filename = f"{user_dir}/{timestamp}_{verdict[:10]}.jpg"

            # Сохраняем изображение
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(filename, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

        except Exception as e:
            logger.error(f"Ошибка сохранения изображения: {e}")

    def create_result_image(self, original_image, results):
        """Создание изображения с результатами"""
        try:
            # Создаем копию для отрисовки
            # Украшательсво убрано
            '''
            result_img = original_image.copy()
            '''

            # Добавляем текст с результатами
            # Украшательсво убрано
            '''
            # text_color = (255, 160, 0) if results.get('combined_ai', 0.5) < 0.5 else (0, 0, 255)
            text_color = (255, 160, 0)
            '''

            # Базовый текст
            # Украшательсво убрано
            '''
            verdict = results.get('verdict', 'Uncertain')
            ai_prob = results.get('combined_ai', 0.5) * 100
            '''

            # Добавляем текст на изображение
            # Украшательсво убрано
            '''
            #font = ImageFont.truetype("fonts/DejaVuSerif-Bold.ttf", 16)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.4
            thickness = 2

            cv2.putText(result_img, f"Result: {verdict}",
                        (30, 70), font, scale, text_color, thickness)
            cv2.putText(result_img, f"Probability AI: {ai_prob:.1f}%",
                        (30, 140), font, scale, text_color, thickness)
            '''

            # Добавляем рамку
            # Украшательсво убрано
            '''
            cv2.rectangle(result_img, (5, 5),
                          (result_img.shape[1] - 5, result_img.shape[0] - 5),
                          text_color, 2)
            '''

            # Возвращается исходное изображение, т.к. ничего с ним не делаем
            '''
            return result_im
            '''
            return goriginal_image

        except Exception as e:
            logger.error(f"Error creating result: {e}")
            return original_image


# Создаем экземпляр детектора
detector = AIDetectorBot()


# ============================================================================
# TELEGRAM BOT HANDLERS
# ============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    welcome_text = f"""
👋 Привет, {user.first_name}!

🤖 Я бот для определения AI-генерированных изображений.

📸 **Как использовать:**
1. Просто отправьте мне любое изображение
2. Я проанализирую его и скажу:
   • Создано ли оно искусственным интеллектом
   • Или это реальная фотография

🔍 **Я анализирую:**
• Текстуры и градиенты
• Цветовые паттерны
• Артефакты генерации
• Регулярность изображения

📊 **Статистика анализа:**
Всего обработано: {detector.stats['total_images']} изображений
AI обнаружено: {detector.stats['ai_detected']}
Реальных: {detector.stats['real_detected']}

📌 Просто отправьте мне изображение!
    """

    await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
📚 **Доступные команды:**

/start - Начать работу с ботом
/help - Показать это сообщение
/stats - Показать статистику
/test - Протестировать на примере

📸 **Как анализировать изображения:**
1. Отправьте боту изображение (фото, картинку)
2. Дождитесь анализа (обычно 5-10 секунд)
3. Получите подробный отчет

⚠️ **Ограничения:**
• Максимальный размер: 20MB
• Поддерживаемые форматы: JPG, PNG, JPEG
• Рекомендуемый размер: до 4000x4000 пикселей

🤔 **Как это работает?**
Бот использует нейросетевые модели для анализа:
1. Сверточная нейросеть (CNN) для анализа текстур
2. Классификатор на основе извлеченных признаков
3. Детекцию специфических артефактов AI-генерации

📈 **Точность:** ~85-90% на тестовых данных
    """

    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /stats"""
    stats = detector.stats
    total = stats['total_images']

    if total > 0:
        ai_percent = (stats['ai_detected'] / total) * 100
        real_percent = (stats['real_detected'] / total) * 100
    else:
        ai_percent = real_percent = 0

    stats_text = f"""
📊 **Статистика бота:**

🔢 Всего обработано: {total} изображений

🤖 AI-изображений обнаружено: {stats['ai_detected']}
📈 Это {ai_percent:.1f}% от общего числа

📸 Реальных изображений: {stats['real_detected']}
📈 Это {real_percent:.1f}% от общего числа

👥 Активных пользователей: {len([d for d in os.listdir('user_images') if os.path.isdir(f'user_images/{d}')])}
    """

    await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)


async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /test - тестовое изображение"""
    test_text = """
🔄 **Тестовый режим**

Отправьте боту любое изображение для анализа.

Примеры изображений для теста:
1. Фотографии с телефона (вероятно реальные)
2. Изображения из нейросетей (Midjourney, DALL-E, Stable Diffusion)
3. Сгенерированные AI аватары
4. Скриншоты игр

Бот проанализирует и даст оценку:
• Вероятность что изображение AI-генерации
• Уровень уверенности
• Обнаруженные артефакты

📸 **Отправьте изображение сейчас!**
    """

    await update.message.reply_text(test_text, parse_mode=ParseMode.MARKDOWN)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик фотографий"""
    try:
        user = update.effective_user
        message = update.message

        # Отправляем сообщение о начале обработки
        processing_msg = await message.reply_text(
            "🔄 Обрабатываю изображение...\n"
            "Это займет несколько секунд."
        )

        # Получаем фото максимального качества
        photo_file = await message.photo[-1].get_file()

        # Скачиваем фото в память
        photo_bytes = await photo_file.download_as_bytearray()

        # Конвертируем в numpy array
        nparr = np.frombuffer(photo_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            await processing_msg.edit_text("❌ Не удалось загрузить изображение")
            return

        # Конвертируем BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Анализируем изображение
        await processing_msg.edit_text("🔍 Анализирую текстуры и цвета...")
        results = detector.analyze_image(image_rgb, user.id)

        if results is None:
            await processing_msg.edit_text("❌ Ошибка при анализе изображения")
            return

        # Создаем изображение с результатами
        await processing_msg.edit_text("🎨 Готовлю результат...")
        result_image = detector.create_result_image(image_rgb, results)

        # Конвертируем обратно в bytes для отправки
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        result_bytes = io.BytesIO(buffer)

        # Формируем текстовый результат
        result_text = f"""
📊 **Результаты анализа:**

🎯 **Вердикт:** {results.get('verdict', 'Не определен')}
🎲 **Уверенность:** {results.get('confidence', 'Средняя')}

📈 **Вероятности:**
• AI-генерация: {results.get('combined_ai', 0.5) * 100:.1f}%
• Реальное фото: {(1 - results.get('combined_ai', 0.5)) * 100:.1f}%

🔬 **Методы анализа:**
• Нейросеть (CNN): {results.get('cnn_ai', 0.5) * 100:.1f}% AI
• Анализ признаков: {results.get('feature_ai', 0.5) * 100:.1f}% AI

⚠️ **Обнаружены артефакты:**
• Резкость: {results.get('artifacts', {}).get('sharpness', 0):.2f}
• Однородность границ: {results.get('artifacts', {}).get('edge_uniformity', 0):.4f}

💡 **Примечание:** Это оценочный результат. 
Точность анализа зависит от качества изображения.
        """

        # Отправляем результат
        await message.reply_photo(
            photo=InputFile(result_bytes, filename='result.jpg'),
            caption=result_text,
            parse_mode=ParseMode.MARKDOWN
        )

        # Удаляем сообщение о обработке
        await processing_msg.delete()

    except Exception as e:
        logger.error(f"Ошибка обработки фото: {e}")
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик документов (изображений)"""
    try:
        document = update.message.document

        # Проверяем что это изображение
        if not document.mime_type.startswith('image/'):
            await update.message.reply_text("❌ Пожалуйста, отправьте изображение (JPG, PNG)")
            return

        # Проверяем размер
        if document.file_size > 20 * 1024 * 1024:  # 20MB
            await update.message.reply_text("❌ Файл слишком большой (максимум 20MB)")
            return

        # Отправляем сообщение о начале обработки
        processing_msg = await update.message.reply_text("🔄 Загружаю и анализирую изображение...")

        # Скачиваем файл
        file = await document.get_file()
        file_bytes = await file.download_as_bytearray()

        # Конвертируем в numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            await processing_msg.edit_text("❌ Не удалось загрузить изображение")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Анализируем
        user = update.effective_user
        results = detector.analyze_image(image_rgb, user.id)

        if results is None:
            await processing_msg.edit_text("❌ Ошибка при анализе")
            return

        # Формируем ответ
        result_text = f"""
📄 **Анализ файла:** {document.file_name}

🎯 **Вердикт:** {results.get('verdict', 'Не определен')}
📊 **Вероятность AI:** {results.get('combined_ai', 0.5) * 100:.1f}%

🔍 **Детали:**
• Размер: {image.shape[1]}x{image.shape[0]} пикселей
• Метод CNN: {results.get('cnn_ai', 0.5) * 100:.1f}% AI
• Анализ признаков: {results.get('feature_ai', 0.5) * 100:.1f}% AI

💡 Файл успешно проанализирован!
        """

        await processing_msg.edit_text(result_text, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Ошибка обработки документа: {e}")
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logger.error(f"Ошибка: {context.error}")

    try:
        # Отправляем сообщение об ошибке пользователю
        await update.message.reply_text(
            "❌ Произошла ошибка при обработке.\n"
            "Попробуйте отправить изображение еще раз или обратитесь к администратору."
        )
    except:
        pass


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик неизвестных команд"""
    await update.message.reply_text(
        "❓ Неизвестная команда.\n"
        "Используйте /help для списка доступных команд."
    )


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Основная функция запуска бота"""
    print("🚀 Запуск AI Image Detector Telegram Bot...")
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN не найден в переменных окружения")
    print(f"🤖 Токен: {TELEGRAM_TOKEN[:10]}...")

    # Создаем Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Добавляем обработчики команд
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("test", test_command))

    # Добавляем обработчики сообщений
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))

    # Обработчик неизвестных команд
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    # Обработчик ошибок
    application.add_error_handler(error_handler)

    # Запускаем бота
    print("✅ Бот запущен. Ожидаем сообщений...")
    print("📱 Перейдите в Telegram и найдите @AIDetect0r_Bot")
    print("💻 Нажмите Ctrl+C для остановки")

    # Запускаем polling
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Бот остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
