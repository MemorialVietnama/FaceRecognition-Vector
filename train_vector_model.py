from database import get_db_connection, log_query
import cv2
import numpy as np
from PIL import Image
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, RandomBrightness
import pickle
from pathlib import Path
from tensorflow.keras.models import save_model

# Получаем путь к директории, где находится скрипт
SCRIPT_DIR = Path(__file__).parent.absolute()


def create_encoder(input_shape=(224, 224, 3)):
    # Аугментация внутри модели (активна только во время обучения)
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
        RandomBrightness(0.1, value_range=(0, 1)),
    ], name="data_augmentation")

    inputs = Input(shape=input_shape)
    x = data_augmentation(inputs)  # Аугментация применяется к входу

    # Улучшенная CNN-архитектура для обработки лиц
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model

def prepare_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT CARD_NUMBER, IMAGE_DATA FROM CLIENT_BIOMETRY"
    cursor.execute(query)
    log_query(query)

    images = []
    labels = []
    processed_images = 0
    failed_images = 0

    print("Обработка записей базы данных...")
    for card_number, image_blob in cursor:
        try:
            if isinstance(image_blob, (bytes, bytearray)):
                image_data = image_blob
            else:
                image_data = image_blob.read()

            if not image_data:
                print(f"Пустые данные изображения для карты {card_number}")
                failed_images += 1
                continue

            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)

            face_image = detect_face(image)
            if face_image is None:
                print(f"Лицо не обнаружено для карты {card_number}")
                failed_images += 1
                continue

            images.append(face_image)
            labels.append(str(card_number))
            processed_images += 1
            print(f"Обработано изображение {processed_images} для карты {card_number}")

        except Exception as e:
            print(f"Ошибка обработки изображения для карты {card_number}: {e}")
            failed_images += 1
            continue

    cursor.close()
    conn.close()
    print(f"Соединение с базой данных закрыто. Обработано: {processed_images}, Ошибок: {failed_images}")

    if not images:
        raise ValueError("Нет валидных изображений, обработанных из базы данных")

    images, labels = shuffle(np.array(images), np.array(labels), random_state=42)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    return images, encoded_labels, label_encoder

def detect_face(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple = (30, 30),
    target_size: tuple = (224, 224),
) -> np.ndarray | None:
    # Кэширование каскада
    if not hasattr(detect_face, "face_cascade"):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        detect_face.face_cascade = cv2.CascadeClassifier(cascade_path)
        if detect_face.face_cascade.empty():
            raise RuntimeError("Не удалось загрузить каскад Хаара")

    # Проверка входных данных
    if image is None or image.size == 0:
        return None

    # Конвертация в серое (для BGR и RGB)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Детекция лиц
    faces = detect_face.face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    if len(faces) == 0:
        return None

    # Выбор лица с максимальной площадью
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Проверка границ изображения
    height, width = image.shape[:2]
    x, y = max(x, 0), max(y, 0)
    w = min(w, width - x)
    h = min(h, height - y)

    # Вырезание лица
    face_image = image[y:y+h, x:x+w]

    # Ресайз до целевого размера
    face_image = cv2.resize(face_image, target_size, interpolation=cv2.INTER_AREA)
    
    return face_image

def process_and_train():
    # Загрузка данных
    all_images, all_labels, label_encoder = prepare_data()
    
    # Предобработка для MobileNetV2
    all_images = preprocess_input(all_images.astype('float32'))
    
    # Создание энкодера
    encoder = create_encoder()
    vectors = encoder.predict(all_images)
    
    # Подготовка данных для сохранения
    card_numbers = label_encoder.inverse_transform(all_labels)
    
    # 1. Сохранение модели
    model_dir = SCRIPT_DIR / 'model'
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / 'encoder.keras'
    encoder.save(str(model_path))
    print(f"Модель сохранена в: {model_path}")
    
    # 2. Сохранение PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    pca_path = model_dir / 'pca.pkl'
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"PCA модель сохранена в: {pca_path}")
    
    # 3. Сохранение LabelEncoder
    label_encoder_path = model_dir / 'label_encoder.pkl'
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"LabelEncoder сохранен в: {label_encoder_path}")
    
    # 4. Сохранение параметров классов (class_params.pca.pkl)
    class_params = {
        'vectors_2d': vectors_2d,
        'all_labels': all_labels,
        'card_numbers': card_numbers
    }
    
    class_params_path = model_dir / 'class_params.pca.pkl'
    with open(class_params_path, 'wb') as f:
        pickle.dump(class_params, f)
    print(f"Параметры классов сохранены в: {class_params_path}")
    
    # 5. Сохранение эмбеддингов (оригинальные векторы)
    embedding_data = {
        'vectors': vectors,
        'labels': all_labels,
        'card_numbers': card_numbers,
        'classes': label_encoder.classes_
    }
    
    embedding_path = model_dir / 'embeddings.pkl'
    with open(embedding_path, 'wb') as f:
        pickle.dump(embedding_data, f)
    print(f"Эмбеддинги сохранены в: {embedding_path}")
    
    # 6. Сохранение векторов по классам
    vectors_by_class = {}
    for cn, vec in zip(card_numbers, vectors):
        vectors_by_class.setdefault(cn, []).append(vec)
    
    for cn, vec_list in vectors_by_class.items():
        try:
            np.save(model_dir / f'{cn}.npy', np.array(vec_list))
            print(f"Сохранен вектор для {cn}")
        except Exception as e:
            print(f"Ошибка сохранения вектора {cn}: {e}")
    
    # Визуализация
    plt.figure(figsize=(20, 15))
    ax = plt.gca()
    scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=all_labels, cmap='jet', alpha=0.5)
    
    # Отображение примеров изображений
    indices = np.random.choice(len(vectors_2d), min(20, len(vectors_2d)), replace=False)
    for idx in indices:
        img = all_images[idx]
        img = (img - img.min()) / (img.max() - img.min())  # Нормализация для отображения
        img = OffsetImage(img, zoom=0.1)
        ax.add_artist(AnnotationBbox(img, vectors_2d[idx], frameon=False))
    
    plt.title('Визуализация векторов лиц')
    
    # Сохранение визуализации
    viz_path = model_dir / 'face_embeddings_visualization.png'
    plt.savefig(str(viz_path))
    print(f"Визуализация сохранена в: {viz_path}")
    plt.show()

if __name__ == "__main__":
    process_and_train()