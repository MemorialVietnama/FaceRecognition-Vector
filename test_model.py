import os
import cv2
import numpy as np
import joblib
import pickle
import tensorflow as tf
from database import get_db_connection
from vector import detect_face, extract_facial_features  # Предполагаем, что эти функции доступны
from PIL import Image
import io
from scipy.spatial.distance import mahalanobis
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные для моделей
_models = {
    'encoder': None,
    'pca': None,
    'label_encoder': None,
    'class_params': {},
    'initialized': False
}

def _initialize_models():
    """Инициализация моделей (выполняется один раз)"""
    if _models['initialized']:
        return
    


    
    try:
        # Определение путей
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, 'flask_app', 'model')
        
        # Проверка наличия файлов
        required_files = {
            'encoder': 'encoder.keras',
            'pca': 'pca.pkl',
            'label_encoder': 'label_encoder.pkl'
        }
        
        missing = []
        for name, file in required_files.items():
            path = os.path.join(model_dir, file)
            if not os.path.exists(path):
                missing.append(path)
        
        if missing:
            raise FileNotFoundError(f"Missing model files: {', '.join(missing)}")

        # Загрузка моделей
        _models['encoder'] = tf.keras.models.load_model(os.path.join(model_dir, 'encoder.keras'))
        _models['pca'] = joblib.load(os.path.join(model_dir, 'pca.pkl'))
        _models['label_encoder'] = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        
        # Загрузка параметров классов из БД
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT CARD_NUMBER, MEAN_VECTOR, COV_MATRIX FROM CLASS_THRESHOLD_REGIONS")
            for row in cursor.fetchall():
                card_number = row[0]
                _models['class_params'][card_number] = {
                    'mean': pickle.loads(row[1]),
                    'cov': pickle.loads(row[2])
                }
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
        
        logger.info("Models initialized successfully")
        logger.info(f"Encoder input shape: {_models['encoder'].input_shape}")
        logger.info(f"PCA components: {_models['pca'].components_.shape}")
        logger.info(f"LabelEncoder classes: {_models['label_encoder'].classes_}")
        logger.info(f"Loaded parameters for {_models['class_params'].shape[0]} classes")
        
        _models['initialized'] = True
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}", exc_info=True)
        raise

def preprocess_image(image_data):
    """Предобработка изображения"""
    try:
        # Обработка изображения из байтового потока
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        
        # Конвертация в RGB, если нужно
        if len(image.shape) == 2:  # Градации серого
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        face_image = detect_face(image)
        if face_image is None:
            raise ValueError("Лицо не обнаружено")
            
        return face_image.astype(np.float32) / 255.0
    except Exception as e:
        raise RuntimeError(f"Ошибка обработки изображения: {str(e)}")

def extract_features(image):
    """Извлечение и объединение признаков"""
    encoded = _models['encoder'].predict(np.expand_dims(image, axis=0))[0]
    facial_features = extract_facial_features((image * 255).astype(np.uint8))
    if facial_features is None:
        raise ValueError("Не удалось извлечь лицевые признаки")
    return np.concatenate([encoded.flatten(), facial_features])

def find_nearest_class(features, threshold=3.0):
    """Поиск ближайшего класса с пороговым расстоянием"""
    min_distance = float('inf')
    best_class = None
    
    # Применение PCA
    pca_vector = _models['pca'].transform([features])[0]
    
    for card_number, params in _models['class_params'].items():
        try:
            cov = params['cov']
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.diag(1/np.diag(cov))
        
        diff = pca_vector - params['mean']
        distance = mahalanobis(pca_vector, params['mean'], inv_cov)
        
        if distance < min_distance and distance <= threshold:
            min_distance = distance
            best_class = card_number
            
    return best_class, min_distance

def recognize_face(image_data):
    """
    Основная функция распознавания
    
    Args:
        image_data: байтовые данные изображения
        
    Returns:
        tuple: (card_number, distance) если найдено лицо
        raises: RuntimeError если лицо не распознано или произошла ошибка
    """
    try:
        # Инициализация моделей при первом вызове
        _initialize_models()
        
        # Предобработка изображения
        processed_image = preprocess_image(image_data)
        
        # Извлечение признаков
        features = extract_features(processed_image)
        
        # Поиск ближайшего класса
        card_number, distance = find_nearest_class(features)
        
        if card_number is None:
            raise RuntimeError("Пользователь не распознан")
            
        return card_number, distance
        
    except Exception as e:
        logger.error(f"Recognition error: {str(e)}", exc_info=True)
        raise