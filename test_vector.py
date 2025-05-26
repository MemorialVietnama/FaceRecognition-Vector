import os
import cv2
import numpy as np
import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.spatial.distance import mahalanobis
from matplotlib import pyplot as plt

# Импорт функций из vector.py
from vector import detect_face, extract_facial_features, calculate_threshold_region

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class FaceRecognizer:
    def __init__(self):
        """Инициализация моделей при создании экземпляра класса"""
        self.encoder = None
        self.pca = None
        self.label_encoder = None
        self.class_params = None  # Изменил имя переменной с class_params_pca на class_params
        self._load_models()
    
    def _load_models(self):
        """Загрузка моделей и данных из файлов с проверкой ошибок"""
        try:
            # Получаем абсолютный путь к директории скрипта
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(base_dir, 'model')
            
            # Формируем полные пути к файлам (исправлены имена файлов)
            model_paths = {
                'encoder': os.path.join(model_dir, 'face_encoder.keras'),  # Изменил encoder.keras на face_encoder.keras
                'pca': os.path.join(model_dir, 'pca.pkl'),
                'label_encoder': os.path.join(model_dir, 'label_encoder.pkl'),
                'class_params': os.path.join(model_dir, 'class_params.pca.pkl')  # Изменил class_params_pca.pkl на class_params.pca.pkl
            }

            # Проверка существования файлов
            missing_files = [name for name, path in model_paths.items() if not os.path.exists(path)]
            if missing_files:
                raise FileNotFoundError(
                    f"Отсутствуют файлы моделей: {', '.join(missing_files)}. "
                    f"Проверьте папку: {model_dir}"
                )

            # Загрузка моделей с обработкой возможных ошибок
            print("⌛ Загрузка моделей...")
            
            # Загрузка encoder
            self.encoder = load_model(model_paths['encoder'])
            print(f"✅ Encoder загружен (Input: {self.encoder.input_shape}, Output: {self.encoder.output_shape})")
            
            # Загрузка PCA
            with open(model_paths['pca'], 'rb') as f:  # Изменил joblib.load на pickle.load
                self.pca = pickle.load(f)
            print(f"✅ PCA загружен (Компоненты: {self.pca.n_components_})")
            
            # Загрузка LabelEncoder
            with open(model_paths['label_encoder'], 'rb') as f:  # Изменил joblib.load на pickle.load
                self.label_encoder = pickle.load(f)
            print(f"✅ LabelEncoder загружен (Классы: {self.label_encoder.classes_.tolist()})")
            
            # Загрузка параметров классов
            with open(model_paths['class_params'], 'rb') as f:
                self.class_params = pickle.load(f)  # Изменил имя переменной на class_params
            print(f"✅ Параметры классов загружены (Количество классов: {len(self.class_params)})")

        except FileNotFoundError as fnf_err:
            print(f"❌ Критическая ошибка: {fnf_err}")
            raise
        except Exception as e:
            print(f"❌ Ошибка при загрузке моделей: {str(e)}")
            print("Проверьте:")
            print("- Версии форматов файлов (например, .keras vs .h5)")
            print("- Целостность файлов моделей")
            print("- Совместимость версий библиотек")
            raise RuntimeError("Невозможно инициализировать модели") from e

    def _process_image(self, image):
        """
        Обработка изображения: детекция лица, извлечение признаков, векторизация
        Args:
            image (np.ndarray): Массив изображения (OpenCV)
        """
        print("\nОбработка изображения...")
        
        # Детекция лица
        face_image = detect_face(image)
        if face_image is None:
            raise ValueError("Лицо не обнаружено на изображении")
        print(f"Размер обнаруженного лица: {face_image.shape}")

        # Приведение к размеру, ожидаемому моделью (224x224)
        if face_image.shape[0] != 224 or face_image.shape[1] != 224:
            print(f"Изменение размера изображения с {face_image.shape} на (224, 224)")
            face_image = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)
        
        original_face_image = face_image.copy()

        # Нормализация и предобработка
        face_image = face_image / 255.0
        face_image = np.expand_dims(face_image, axis=0)  # Добавляем batch dimension

        # Проверка формы перед подачей в модель
        if face_image.shape != (1, 224, 224, 3):
            raise ValueError(f"Некорректная форма изображения для модели. Ожидается (1, 224, 224, 3), получено {face_image.shape}")
        
        # Векторизация
        encoded_vector = self.encoder.predict(face_image, verbose=0)[0]
        print(f"Вектор после encoder: {encoded_vector.shape}")

        facial_features = extract_facial_features((face_image[0] * 255).astype(np.uint8))
        if facial_features is None:
            raise ValueError("Не удалось извлечь признаки лица")
        print(f"Признаки лица: {facial_features.shape}")

        combined_vector = np.concatenate([encoded_vector.flatten(), facial_features])
        print(f"Объединенный вектор: {combined_vector.shape}")

        combined_pca = self.pca.transform([combined_vector])
        print(f"Вектор после PCA: {combined_pca.shape}")

        return combined_pca[0], original_face_image

    def _check_match(self, query_vector, threshold_factor=1.3):
        """Проверка попадания вектора в пороговые области"""
        print("\nПроверка совпадения...")
        results = []

        for class_name, params in self.class_params.items():
            mean = params['mean']
            cov = params['cov']

            inv_cov = np.linalg.inv(cov)
            distance = mahalanobis(query_vector, mean, inv_cov)  # оба (2,)

            region_params = calculate_threshold_region(cov, sigma=3.0)
            threshold_distance = np.sqrt(region_params['major_axis']**2 + region_params['minor_axis']**2) * threshold_factor

            confidence = max(0, 100 - (distance / threshold_distance) * 100) if distance <= threshold_distance else 0
            confidence = round(confidence, 2)

            matched = bool(distance <= threshold_distance)  # Явное преобразование в bool
            results.append({
                'class': str(class_name),  # Убедитесь, что строка
                'distance': float(distance),  # Преобразование в float
                'threshold': float(threshold_distance),
                'matched': matched,  # Теперь это bool, а не numpy.bool_
                'confidence': float(confidence)
            })

        results.sort(key=lambda x: x['distance'])
        return results

    def _show_result(self, face_image, results):
        """Отображение результата с изображением и уверенностью"""
        plt.figure(figsize=(10, 6))
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        plt.imshow(face_image_rgb)
        plt.axis('off')

        title_text = ""
        matched_class = next((r for r in results if r['matched']), None)
        if matched_class:
            title_text = f"✅ Точное совпадение: {matched_class['class']} ({matched_class['confidence']}%)"
            plt.title(title_text, fontsize=14, color='green')
        else:
            closest = results[0]
            title_text = f"⚠️ Ближайший класс: {closest['class']} ({closest['confidence']}%)"
            plt.title(title_text, fontsize=14, color='orange')

        print("\n🔍 Результаты:")
        for i, res in enumerate(results):
            matched_str = "✅ Совпадает" if res['matched'] else "❌ Не совпадает"
            conf = res['confidence']
            dist = res['distance']
            print(f"{i+1}. {res['class']}: Расстояние = {dist:.4f}, Уверенность = {conf:.2f}%, {matched_str}")

        plt.show()
    
    def recognize(self, image):
        """
        Основная функция распознавания лица
        Args:
            image (np.ndarray): Массив изображения (OpenCV)
        """
        try:
            
            # Обработка изображения
            query_vector, face_image = self._process_image(image)

            # Проверка совпадения
            results = self._check_match(query_vector)
            
            # Возвращаем результаты для возможного дальнейшего использования
            return results
            
        except Exception as e:
            print(f"Ошибка при распознавании: {str(e)}")
            return None
