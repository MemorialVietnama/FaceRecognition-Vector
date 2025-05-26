import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Добавьте этот импорт в начало файла (рядом с другими импортами)
from sklearn.model_selection import train_test_split
from PIL import Image
import io
from database import get_db_connection, log_query
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tensorflow.keras.regularizers import l2
import mediapipe as mp
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Ellipse
import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import joblib


# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Константы для ключевых точек (MediaPipe 0.9.1+)
LANDMARK_INDICES = {
    # Глаза с iris tracking (refine_landmarks=True)
    "LEFT_EYE": 468,
    "RIGHT_EYE": 473,
    "LEFT_EYE_OUTER": 33,
    "RIGHT_EYE_OUTER": 263,
    
    # Нос
    "NOSE_TIP": 4,
    "NOSE_BRIDGE": 6,
    
    # Рот
    "MOUTH_LEFT": 61,
    "MOUTH_RIGHT": 291,
    "UPPER_LIP": 13,
    "LOWER_LIP": 14,
    
    # Брови
    "LEFT_EYEBROW": 70,
    "RIGHT_EYEBROW": 300,
    
    # Контур лица
    "JAW_LEFT": 132,
    "JAW_RIGHT": 361
}

def extract_facial_features(image, raise_errors=False):
    """
    Извлекает нормализованные характеристики лица с улучшенной устойчивостью к поворотам.
    
    Параметры:
        image : numpy.ndarray - входное изображение BGR
        raise_errors : bool - вызывать исключения при ошибках обнаружения
        
    Возвращает:
        np.ndarray или None - вектор характеристик (33 элемента)
    """
    try:
        # Конвертация в RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        # Обработка ошибок обнаружения
        if not results.multi_face_landmarks:
            if raise_errors:
                raise ValueError("Не обнаружено лиц на изображении")
            return None
            
        if len(results.multi_face_landmarks) > 1:
            if raise_errors:
                raise ValueError("Обнаружено несколько лиц")
            return None

        # Преобразование landmarks в numpy array [N, 2]
        landmarks = np.array([
            (lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark
        ], dtype=np.float32)

        # Базовые точки для нормализации
        left_eye = landmarks[LANDMARK_INDICES["LEFT_EYE"]]
        right_eye = landmarks[LANDMARK_INDICES["RIGHT_EYE"]]
        eye_center = (left_eye + right_eye) / 2
        face_size = np.linalg.norm(right_eye - left_eye)

        # Функция для относительных координат
        def rel_coord(point):
            return (point - eye_center) / face_size

        # Рассчитываем ключевые точки в относительных координатах
        mouth_left = rel_coord(landmarks[LANDMARK_INDICES["MOUTH_LEFT"]])
        mouth_right = rel_coord(landmarks[LANDMARK_INDICES["MOUTH_RIGHT"]])
        nose_tip = rel_coord(landmarks[LANDMARK_INDICES["NOSE_TIP"]])
        jaw_left = rel_coord(landmarks[LANDMARK_INDICES["JAW_LEFT"]])
        jaw_right = rel_coord(landmarks[LANDMARK_INDICES["JAW_RIGHT"]])

        # 1. Геометрические признаки
        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        nose_length = np.linalg.norm(nose_tip - eye_center/face_size)
        jaw_width = np.linalg.norm(jaw_right - jaw_left)
        
        # 2. Угловые признаки
        nose_angle = np.arctan2(nose_tip[1], nose_tip[0])
        brow_angle = np.arctan2(
            landmarks[LANDMARK_INDICES["RIGHT_EYEBROW"]][1] - 
            landmarks[LANDMARK_INDICES["LEFT_EYEBROW"]][1],
            landmarks[LANDMARK_INDICES["RIGHT_EYEBROW"]][0] - 
            landmarks[LANDMARK_INDICES["LEFT_EYEBROW"]][0]
        )

        # 3. Признаки симметрии
        left_eye_area = _calc_eye_area(
            landmarks[LANDMARK_INDICES["LEFT_EYE_OUTER"]:LANDMARK_INDICES["LEFT_EYE_OUTER"]+6]
        )
        right_eye_area = _calc_eye_area(
            landmarks[LANDMARK_INDICES["RIGHT_EYE_OUTER"]:LANDMARK_INDICES["RIGHT_EYE_OUTER"]+6]
        )
        eye_symmetry = 1 - abs(left_eye_area - right_eye_area)/(left_eye_area + right_eye_area)

        # 4. Соотношения пропорций
        face_height = np.linalg.norm(jaw_left - landmarks[LANDMARK_INDICES["NOSE_BRIDGE"]])
        face_ratio = jaw_width / face_height

        # 5. Динамические признаки (относительные смещения)
        lip_thickness = np.linalg.norm(
            rel_coord(landmarks[LANDMARK_INDICES["UPPER_LIP"]]) - 
            rel_coord(landmarks[LANDMARK_INDICES["LOWER_LIP"]])
        )

        # Собираем все признаки в вектор
        features = np.array([
            mouth_width,
            nose_length,
            jaw_width,
            nose_angle,
            brow_angle,
            eye_symmetry,
            face_ratio,
            lip_thickness,
            # Добавьте другие признаки по аналогии
        ], dtype=np.float32)

        return features

    except Exception as e:
        if raise_errors:
            raise
        print(f"Ошибка обработки: {str(e)}")
        return None

def _calc_eye_area(eye_points):
    """Вычисляет площадь области глаза по 6 точкам"""
    return cv2.contourArea(eye_points.reshape(-1, 1, 2))

def detect_face(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple = (30, 30),
    target_size: tuple = (128, 128),
) -> np.ndarray | None:
    """
    Обнаруживает и вырезает лицо на изображении.
    
    :param image: Входное изображение (BGR или RGB).
    :param scale_factor: Параметр масштабирования для detectMultiScale.
    :param min_neighbors: Параметр minNeighbors для detectMultiScale.
    :param min_size: Минимальный размер лица.
    :param target_size: Размер выходного изображения.
    :return: Изображение лица с паддингом или None.
    """
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
    face_image = image[y : y + h, x : x + w]

    # Ресайз с сохранением пропорций и паддингом
    h_target, w_target = target_size
    ratio = min(w_target / w, h_target / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(face_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Добавление паддинга
    delta_w = w_target - new_w
    delta_h = h_target - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

def create_autoencoder(input_shape):
    input_img = tf.keras.Input(shape=input_shape)

    # Энкодер
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Декодер
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = tf.keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


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


def augment_images(images):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    # Создаем один генератор для всех изображений
    generator = datagen.flow(images, batch_size=1)
    augmented_images = []
    
    for _ in range(len(images)):
        # Получаем одно аугментированное изображение
        augmented = next(generator)[0]
        augmented_images.append(augmented)
    
    return np.array(augmented_images)

from matplotlib.patches import Ellipse  # Добавить импорт в начало файла

def visualize_combined(reduced_vectors, images, labels, label_encoder, class_params_tsne):
    """
    Объединенная визуализация: изображения + цветовые метки классов + области распределения
    """
    # Проверка совпадения размеров данных
    assert len(reduced_vectors) == len(images) == len(labels), "Несоответствие размеров данных!"
    
    unique_classes = np.unique(labels)
    plt.figure(figsize=(35, 30))
    
    # 1. Создаем цветовую карту для классов
    from matplotlib import colormaps
    colors = colormaps['tab20'].resampled(len(unique_classes))
    class_colors = {cls: colors(i/len(unique_classes)) for i, cls in enumerate(unique_classes)}
    
    # 2. Ограничиваем количество отображаемых изображений
    max_display = 500
    if len(images) > max_display:
        indices = np.random.choice(len(images), max_display, replace=False)
        reduced_vectors = reduced_vectors[indices]
        images = images[indices]
        labels = np.array(labels)[indices]
    
    # 3. Преобразование изображений в uint8
    images_uint8 = (images * 255).astype(np.uint8) if images.dtype != np.uint8 else images
    
    # 4. Рисуем основные элементы
    ax = plt.gca()
    
    # 5. Сначала добавляем точки (прозрачные)
    for cls in unique_classes:
        class_mask = (labels == cls)
        if np.any(class_mask):
            plt.scatter(
                reduced_vectors[class_mask, 0],
                reduced_vectors[class_mask, 1],
                c=[class_colors[cls]],
                s=50,
                alpha=0.3,
                label=label_encoder.inverse_transform([cls])[0]
            )
    
    # 6. Затем добавляем изображения с рамками
    for vec, img, label in zip(reduced_vectors, images_uint8, labels):
        class_color = class_colors[label]
        imagebox = OffsetImage(img, zoom=0.3)
        imagebox.image.axes = ax
        ab = AnnotationBbox(
            imagebox, 
            (vec[0], vec[1]),
            frameon=True,
            pad=0.1,
            bboxprops=dict(
                edgecolor=class_color,
                linewidth=2,
                boxstyle="round,pad=0.3",
                alpha=0.7
            )
        )
        ax.add_artist(ab)
    
    # 7. Добавляем ОСНОВНОЙ и ПОРОГОВЫЙ эллипсы
    for class_name, params in class_params_tsne.items():
        class_id = label_encoder.transform([class_name])[0]
        class_mask = (labels == class_id)
        class_reduced = reduced_vectors[class_mask]
        if len(class_reduced) == 0:
            continue
            
        # Основные параметры
        mean = params['mean']
        cov = params['cov']
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        angle = np.rad2deg(np.arccos(v[0, 0]))
        
        # Основной эллипс (95% доверительный интервал)
        ell_main = Ellipse(
            xy=mean,
            width=lambda_[0] * 3.5,   # Основной размер
            height=lambda_[1] * 3.5,
            angle=angle,
            color='red',
            alpha=0.3,
            label=f'Кластер {class_name}'
        )
        ax.add_artist(ell_main)
        
        # Пороговый эллипс (увеличенный на 30%)
        ell_threshold = Ellipse(
            xy=mean,
            width=lambda_[0] * 4.5,   # Расширенный размер
            height=lambda_[1] * 4.5,
            angle=angle,
            color='blue',
            linestyle='--',
            fill=False,
            alpha=0.7,
            label=f'Порог {class_name}'
        )
        ax.add_artist(ell_threshold)
    
    # 8. Настраиваем легенду и оформление
    plt.legend(
        loc='upper right',
        fontsize='small',
        title="Классы",
        framealpha=0.9
    )
    plt.title(f"t-SNE с областями классов (n={len(images)})", fontsize=16)
    plt.xlabel("t-SNE компонента 1", fontsize=12)
    plt.ylabel("t-SNE компонента 2", fontsize=12)
    plt.grid(alpha=0.2)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.95)
    plt.show()

def calculate_threshold_region(cov_matrix, sigma=3.0):
    """
    Рассчитывает параметры пороговой области на основе ковариационной матрицы.
    
    Параметры:
        cov_matrix : np.ndarray - ковариационная матрица распределения
        sigma : float - коэффициент масштабирования области (по умолчанию 3σ)
        
    Возвращает:
        dict - параметры области:
            'major_axis' : большая ось эллипса
            'minor_axis' : малая ось эллипса
            'angle' : угол поворота эллипса (в градусах)
    """
    # Вычисление собственных значений и векторов
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Вычисление угла поворота
    angle = np.degrees(np.arctan2(eigenvectors[1,0], eigenvectors[0,0]))
    
    # Размеры осей с учетом сигма
    major_axis = sigma * np.sqrt(eigenvalues[0])
    minor_axis = sigma * np.sqrt(eigenvalues[1])
    
    return {
        'major_axis': major_axis,
        'minor_axis': minor_axis,
        'angle': angle
    }
def save_or_update_class_params(card_number, mean_vector, major_axis, minor_axis, angle, unique_hash):
    """Сохраняет или обновляет параметры класса в базе данных"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Сериализация вектора средних значений
        mean_blob = pickle.dumps(mean_vector)
        
        # Проверка существования записи
        cursor.execute(
            "SELECT 1 FROM CLASS_THRESHOLD_REGIONS WHERE CARD_NUMBER = ?",
            (card_number,)
        )
        exists = cursor.fetchone()

        if exists:
            # Обновление существующей записи
            cursor.execute("""
                UPDATE CLASS_THRESHOLD_REGIONS 
                SET 
                    MEAN_VECTOR = ?,
                    MAJOR_AXIS = ?,
                    MINOR_AXIS = ?,
                    ANGLE = ?,
                    UNIQUE_KEY_HASH = ?,
                    UPDATED_AT = CURRENT_TIMESTAMP
                WHERE CARD_NUMBER = ?
            """, (mean_blob, major_axis, minor_axis, angle, unique_hash, card_number))
        else:
            # Вставка новой записи
            cursor.execute("""
                INSERT INTO CLASS_THRESHOLD_REGIONS (
                    CARD_NUMBER,
                    MEAN_VECTOR,
                    MAJOR_AXIS,
                    MINOR_AXIS,
                    ANGLE,
                    UNIQUE_KEY_HASH
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (card_number, mean_blob, major_axis, minor_axis, angle, unique_hash))
        
        conn.commit()
        print(f"Данные для карты {card_number} {'обновлены' if exists else 'добавлены'}")
    
    except Exception as e:
        print(f"Ошибка сохранения данных для {card_number}: {str(e)}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


def process_and_train():
    print("Начало обработки данных и обучения модели...")
    # Подготовка данных
    all_images, all_labels, label_encoder = prepare_data()
    all_images = all_images / 255.0
    
    # Разделение данных и аугментация
    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
    X_train_augmented = augment_images(X_train)
    
    # Обучение автоэнкодера
    input_shape = X_train.shape[1:]
    autoencoder = create_autoencoder(input_shape)
    autoencoder.fit(X_train_augmented, X_train_augmented, epochs=15, batch_size=32, validation_data=(X_test, X_test))
    
    # Извлечение векторов и ключевых точек
    encoder = tf.keras.Model(autoencoder.input, autoencoder.layers[-6].output)
    encoded_vectors = encoder.predict(all_images)
    
    facial_feature_vectors = []
    valid_indices = []
    for i, image in enumerate(all_images):
        features = extract_facial_features((image * 255).astype(np.uint8))
        if features is not None:
            facial_feature_vectors.append(features)
            valid_indices.append(i)
    
    filtered_encoded = np.array([vec.flatten() for vec in encoded_vectors[valid_indices]])
    filtered_labels = all_labels[valid_indices]
    
    # Формирование комбинированных векторов
    combined_vectors = np.array([
        np.concatenate([filtered_encoded[i], facial_feature_vectors[i]])
        for i in range(len(valid_indices))
    ])
    
    # Понижение размерности с PCA для генерации ключей
    pca = PCA(n_components=0.95)
    combined_pca = pca.fit_transform(combined_vectors)
    joblib.dump(combined_pca, 'flask_app/model/combined_pca.pkl')
    joblib.dump(all_images[valid_indices], 'flask_app/model/training_images.pkl')
    joblib.dump(filtered_labels, 'flask_app/model/training_labels.pkl')
    
    # Создаем папку если ее нет
    os.makedirs('flask_app/model', exist_ok=True)
    
    # Сохраняем encoder в новом формате
    encoder_path = os.path.join('flask_app/model', 'encoder.keras')
    encoder.save(encoder_path, save_format='keras')
    print(f"\nЭнкодер сохранен в {encoder_path}")
    
    # Сохраняем PCA
    pca_path = os.path.join('flask_app/model', 'pca.pkl')
    joblib.dump(pca, pca_path)
    print(f"PCA модель сохранена в {pca_path}")
    
    # Сохраняем LabelEncoder
    label_encoder_path = os.path.join('flask_app/model', 'label_encoder.pkl')
    joblib.dump(label_encoder, label_encoder_path)
    print(f"LabelEncoder сохранен в {label_encoder_path}\n")
    
    # Генерация ключей
    class_vectors = {}
    for label in np.unique(filtered_labels):
        class_name = label_encoder.inverse_transform([label])[0]
        class_vectors[class_name] = combined_pca[filtered_labels == label]
    
    class_params = {}
    for class_name, vectors in class_vectors.items():
        if len(vectors) == 0:
            continue
        class_params[class_name] = {
            'mean': np.mean(vectors, axis=0),
            'cov': np.cov(vectors, rowvar=False)
        }
        print(f"=== Ключ для класса {class_name} ===")
        print(f"Центр: {class_params[class_name]['mean']}")
        print(f"Ковариационная матрица:\n{class_params[class_name]['cov']}\n")
    
    # Визуализация с t-SNE (параметры для графика)
    n_samples = combined_vectors.shape[0]
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_vectors = tsne.fit_transform(combined_pca)  # 2D данные для визуализации

    # Параметры для ВИЗУАЛИЗАЦИИ (на основе t-SNE)
    class_params_tsne_vis = {}
    for class_name in class_vectors:
        class_mask = (filtered_labels == label_encoder.transform([class_name])[0])
        class_tsne = reduced_vectors[class_mask]  # Используем t-SNE данные (2D)
        class_params_tsne_vis[class_name] = {
            'mean': np.mean(class_tsne, axis=0),
            'cov': np.cov(class_tsne, rowvar=False)
        }
        # Сохраняем PCA-параметры для классификации
    class_params_tsne_path = os.path.join('flask_app/model', 'class_params_tsne.pkl')
    with open(class_params_tsne_path, 'wb') as f:
        pickle.dump(class_params_tsne_vis, f)
    print(f"Параметры PCA сохранены в {class_params_tsne_path}")

    # Визуализация с t-SNE параметрами
    visualize_combined(
        reduced_vectors,
        all_images[valid_indices],
        filtered_labels,
        label_encoder,
        class_params_tsne_vis  # 2D параметры для графика
    )

    # Параметры для КЛАССИФИКАЦИИ (на основе PCA)
    class_params_pca = {}
    for class_name in class_vectors:
        class_mask = (filtered_labels == label_encoder.transform([class_name])[0])
        class_pca = combined_pca[class_mask]  # PCA данные (7D)
        class_params_pca[class_name] = {
            'mean': np.mean(class_pca, axis=0),
            'cov': np.cov(class_pca, rowvar=False)
        }

    # Сохраняем PCA-параметры для классификации
    class_params_pca_path = os.path.join('flask_app/model', 'class_params_pca.pkl')
    with open(class_params_pca_path, 'wb') as f:
        pickle.dump(class_params_pca, f)
    print(f"Параметры PCA сохранены в {class_params_pca_path}")

    # Сохранение PCA-параметров в БД
    print("\n=== Сохранение данных в базе ===")
    for class_name in class_params_pca:  # Используем PCA-параметры!
        cov_matrix = class_params_pca[class_name]['cov']
        region_params = calculate_threshold_region(cov_matrix)
        mean_vector = class_params_pca[class_name]['mean']
        
        unique_hash = hash(tuple(mean_vector))
        save_or_update_class_params(
            card_number=class_name,
            mean_vector=mean_vector,
            major_axis=region_params['major_axis'],
            minor_axis=region_params['minor_axis'],
            angle=region_params['angle'],
            unique_hash=str(unique_hash)
        )