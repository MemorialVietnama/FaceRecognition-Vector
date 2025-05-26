import os
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from typing import List, Dict, Tuple

class PhotoValidator:
    def __init__(self):
        self.detector = MTCNN()
        self.min_face_size = 80  # минимальный размер лица в пикселях
        self.min_similarity = 0.6  # минимальное сходство между лицами

    def validate_photos(self, photo_paths: List[str]) -> Dict:
        """
        Основной метод для проверки фотографий
        Возвращает словарь с результатами проверки
        """
        results = {
            'valid': [],
            'invalid': [],
            'all_same_person': False,
            'faces_detected': 0,
            'validation_passed': False
        }

        # Проверяем каждое фото
        face_encodings = []
        for path in photo_paths:
            result = self._validate_single_photo(path)
            if result['has_face']:
                results['valid'].append(result)
                face_encodings.append(result['face_encoding'])
            else:
                results['invalid'].append(result)

        results['faces_detected'] = len(results['valid'])

        # Проверяем, что все лица принадлежат одному человеку
        if len(face_encodings) >= 2:
            results['all_same_person'] = self._check_same_person(face_encodings)

        # Финальная проверка - минимум 3 фото с одним человеком
        if (len(results['valid']) >= 3 and results['all_same_person']):
            results['validation_passed'] = True

        return results

    def _validate_single_photo(self, photo_path: str) -> Dict:
        """
        Проверяет одно фото на наличие лица
        """
        result = {
            'path': photo_path,
            'has_face': False,
            'face_encoding': None,
            'face_location': None,
            'error': None
        }

        try:
            # Загружаем изображение
            image = cv2.cvtColor(cv2.imread(photo_path), cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]

            # Обнаруживаем лица
            faces = self.detector.detect_faces(image)
            
            if not faces:
                result['error'] = "Лицо не обнаружено"
                return result

            # Берем самое большое лицо
            main_face = max(faces, key=lambda x: x['confidence'])

            # Проверяем размер лица
            x, y, w, h = main_face['box']
            if w < self.min_face_size or h < self.min_face_size:
                result['error'] = f"Лицо слишком маленькое ({w}x{h}px)"
                return result

            # Вырезаем область лица
            face_img = image[y:y+h, x:x+w]
            
            # Сохраняем результаты
            result['has_face'] = True
            result['face_location'] = (x, y, w, h)
            result['face_encoding'] = self._get_face_encoding(face_img)

        except Exception as e:
            result['error'] = f"Ошибка обработки: {str(e)}"

        return result

    def _get_face_encoding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Генерирует embedding для лица (упрощенная версия)
        """
        # В реальной реализации используйте нормальный метод получения эмбеддингов
        # Например, FaceNet или другой нейросетевой подход
        face_img = cv2.resize(face_img, (160, 160))
        return face_img.flatten() / 255.0  # упрощенный "эмбеддинг"

    def _check_same_person(self, encodings: List[np.ndarray]) -> bool:
        """
        Проверяет, что все лица принадлежат одному человеку
        """
        # Сравниваем первое лицо с остальными
        base_encoding = encodings[0]
        for encoding in encodings[1:]:
            similarity = self._cosine_similarity(base_encoding, encoding)
            if similarity < self.min_similarity:
                return False
        return True

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Вычисляет косинусную схожесть между векторами
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
