import os
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Архитектура нейросети

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
            nn.Linear(128, 2), # 2 класса: Real (0) и AI (1)
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# 2. Получение свойств

def extract_features(image_array):
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
        return np.zeros(19) # 3 scalar + 16 hist bins

# 3. Процесс обучения
def train_and_save():
    os.makedirs('models', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Устройство для обучения: {device}")

    # --- Настройки ---
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 15
    DATASET_PATH = 'dataset' # Ожидается структура: dataset/train/real и dataset/train/ai - туда закидываем ПРИМЕРЫ

    # Трансформации
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cnn_model = SimpleCNN().to(device)
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    scaler = StandardScaler()

    # --- Проверка наличия данных ---
    has_data = os.path.exists(DATASET_PATH) and os.path.exists(os.path.join(DATASET_PATH, 'train'))

    if has_data:
        print("📂 Найден датасет. Начинаю реальное обучение...")

        # 1. Обучение CNN
        train_dataset = ImageFolder(root=os.path.join(DATASET_PATH, 'train'), transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

        cnn_model.train()
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = cnn_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

        print("📊 Обучение Random Forest на признаках...")
        X_features = []
        y_labels = []

        # Проходим по тем же картинкам для извлечения признаков opencv
        for class_name in ['real', 'ai']:
            class_dir = os.path.join(DATASET_PATH, 'train', class_name)
            #label = 0 if class_name == 'real' else 1
            label = 1 if class_name == 'real' else 0
            if not os.path.exists(class_dir): continue

            for img_name in os.listdir(class_dir):
                try:
                    img_type = "__ AI __ "
                    if label == 1: img_type = "__REAL__ "
                    img_path = os.path.join(class_dir, img_name)
                    logger.info(f"Training on {img_type} image: {img_path}")
                    img = cv2.imread(img_path)
                    if img is None: continue
                    feats = extract_features(img)
                    X_features.append(feats)
                    y_labels.append(label)
                except: pass

        if X_features:
            X_features = np.array(X_features)
            X_features = scaler.fit_transform(X_features)
            rf_classifier.fit(X_features, y_labels)

    else:
        print("⚠️ Датасет не найден. Положите примеры!")

    # --- Сохранение ---
    print("💾 Сохранение моделей...")
    torch.save(cnn_model.state_dict(), 'models/cnn_model.pth')
    joblib.dump(rf_classifier, 'models/classifier.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    print("✅ Все модели успешно сохранены в папку models/!")

if __name__ == "__main__":
    train_and_save()
