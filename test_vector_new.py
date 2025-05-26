from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle
import numpy as np

# Получаем путь к директории со скриптом и моделями
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_DIR = SCRIPT_DIR / 'model'

# Загрузка моделей и данных при старте
encoder = tf.keras.models.load_model(MODEL_DIR / 'encoder.keras')

with open(MODEL_DIR / 'pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open(MODEL_DIR / 'class_params.pca.pkl', 'rb') as f:
    class_params = pickle.load(f)
    vectors_2d = class_params['vectors_2d']
    card_numbers = class_params['card_numbers']

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

    # Конвертация в серое
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

    # Выбор наибольшего лица
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Обрезка и ресайз
    face_image = image[y:y+h, x:x+w]
    return cv2.resize(face_image, target_size, interpolation=cv2.INTER_AREA)

def preprocess_image(image_path: str) -> np.ndarray | None:
    # Загрузка и конвертация в RGB
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Ошибка загрузки изображения: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Детекция и обработка лица
    face_image = detect_face(image)
    if face_image is None:
        return None
    
    # Предобработка для модели
    face_image = preprocess_input(face_image.astype('float32'))
    return np.expand_dims(face_image, axis=0)  # Добавление размерности батча

def recognize(image_path: str) -> str:
    # Предобработка изображения
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return "Лицо не найдено (не удалось обнаружить лицо на изображении)"
    
    # Получение эмбеддинга
    embedding = encoder.predict(processed_image, verbose=0)
    
    # Применение PCA
    new_point = pca.transform(embedding)
    
    # Поиск ближайших соседей
    distances = np.linalg.norm(vectors_2d - new_point, axis=1)
    sorted_indices = np.argsort(distances)
    top3_indices = sorted_indices[:3]
    
    # Сбор информации о топ-3 совпадениях
    top3_info = []
    for idx in top3_indices:
        card = card_numbers[idx]
        dist = distances[idx]
        top3_info.append(f"{card} ({dist:.1f})")
    
    min_distance = distances[top3_indices[0]]
    main_card = card_numbers[top3_indices[0]]
    
    # Формирование результата
    if min_distance < 80:
        return (
            f"Уверенное совпадение: карточка {main_card}\n"
            f"Расстояние: {min_distance:.1f} (порог <80)\n"
            f"Топ-3 совпадений: {', '.join(top3_info)}"
        )
    elif min_distance < 100:
        return (
            f"Частичное совпадение: карточка {main_card}\n"
            f"Расстояние: {min_distance:.1f} (порог 80-100)\n"
            f"Рекомендуется ручная проверка. Топ-3: {', '.join(top3_info)}"
        )
    else:
        return (
            f"Лицо не найдено в базе\n"
            f"Минимальное расстояние: {min_distance:.1f} (порог ≥100)\n"
            f"Ближайшие кандидаты: {', '.join(top3_info)}"
        )
