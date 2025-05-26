import os
import numpy as np
import tensorflow as tf
import cv2  # Добавляем OpenCV для обнаружения лиц
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from PIL import Image
import io
from database import get_db_connection, log_query
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.regularizers import l2

def plot_sample_images(images, labels, label_encoder, num_samples_per_class=5):
    """
    Отображает по несколько изображений каждого класса.
    :param images: Массив изображений.
    :param labels: Массив меток.
    :param label_encoder: Объект LabelEncoder для декодирования меток.
    :param num_samples_per_class: Количество изображений на класс.
    """
    unique_classes = np.unique(labels)
    fig, axes = plt.subplots(len(unique_classes), num_samples_per_class, figsize=(15, 3 * len(unique_classes)))
    
    for i, class_id in enumerate(unique_classes):
        class_name = label_encoder.inverse_transform([class_id])[0]
        class_indices = np.where(labels == class_id)[0]
        selected_indices = np.random.choice(class_indices, size=min(num_samples_per_class, len(class_indices)), replace=False)
        
        for j in range(num_samples_per_class):
            ax = axes[i][j] if len(unique_classes) > 1 else axes[j]
            if j < len(selected_indices):
                idx = selected_indices[j]
                ax.imshow(images[idx])
                ax.set_title(f"Class: {class_name}")
            else:
                ax.axis('off')  # Если изображений меньше, чем требуется, оставляем пустые ячейки
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()



def plot_predictions(model, images, true_labels, label_encoder, num_samples=9):
    """
    Отображает оригинальные изображения и их предсказанные метки.
    :param model: Обученная модель.
    :param images: Массив изображений.
    :param true_labels: Массив истинных меток.
    :param label_encoder: Объект LabelEncoder для декодирования меток.
    :param num_samples: Количество изображений для отображения.
    """
    predictions = model.predict(images[:num_samples])
    predicted_labels = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i])
        true_label = label_encoder.inverse_transform([true_labels[i]])[0]
        predicted_label = label_encoder.inverse_transform([predicted_labels[i]])[0]
        ax.set_title(f"True: {true_label}\nPredicted: {predicted_label}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def detect_face(image):
    """
    Обнаруживает лицо на изображении с помощью OpenCV.
    :param image: Исходное изображение (numpy array).
    :return: Изображение с вырезанным лицом (numpy array) или None, если лицо не найдено.
    """
    # Загрузка предварительно обученного классификатора Haar Cascade для обнаружения лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Преобразование изображения в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("Лицо не обнаружено.")
        return None
    
    # Вырезаем первое обнаруженное лицо
    x, y, w, h = faces[0]
    face_image = image[y:y+h, x:x+w]
    
    # Изменяем размер лица до целевого размера (128x128)
    face_image = cv2.resize(face_image, (128, 128))
    
    return face_image


def augment_image(image, augmentation_model, num_augmentations=10):
    """
    Применяет аугментацию к одному изображению и возвращает список аугментированных изображений.
    :param image: Исходное изображение (numpy array).
    :param augmentation_model: Модель аугментации (Sequential модель TensorFlow).
    :param num_augmentations: Количество аугментированных изображений для создания.
    :return: Список аугментированных изображений.
    """
    # Преобразование изображения в тензор tf.Tensor
    image_tensor = tf.expand_dims(image, axis=0)
    
    # Применение аугментации
    augmented_images = [augmentation_model(image_tensor)[0].numpy() for _ in range(num_augmentations)]
    
    # Ограничение значений пикселей в диапазоне [0, 1]
    augmented_images = [np.clip(img, 0, 1) for img in augmented_images]
    
    # Нормализация (если требуется)
    augmented_images = [img / 255.0 if img.max() > 1 else img for img in augmented_images]
    
    return augmented_images




def create_dataset_and_train_model():
    print("Начало создания датасета...")
    
    # Подключение к базе данных
    print("Подключение к базе данных...")
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT CARD_NUMBER, IMAGE_DATA FROM CLIENT_BIOMETRY"
    print(f"Выполнение запроса: {query}")
    cursor.execute(query)
    log_query(query)
    
    images = []  # Для изображений с лицами
    labels = []
    processed_images = 0
    failed_images = 0
    
    print("Обработка записей базы данных...")
    for card_number, image_blob in cursor:
        try:
            # Чтение данных изображения
            if isinstance(image_blob, (bytes, bytearray)):
                image_data = image_blob
            else:
                image_data = image_blob.read()
            
            if not image_data:
                print(f"Пустые данные изображения для карты {card_number}")
                failed_images += 1
                continue
            
            # Преобразование в RGB
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                print(f"Конвертация изображения для карты {card_number} в RGB")
                image = image.convert('RGB')
                
            image = np.array(image)  # Преобразуем в numpy array для OpenCV
            
            # Обнаружение лица
            face_image = detect_face(image)
            if face_image is None:
                print(f"Лицо не обнаружено для карты {card_number}")
                failed_images += 1
                continue
            
            # Добавление изображения с лицом в датасет
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
    
    # Перемешивание данных
    all_images, all_labels = shuffle(np.array(images), np.array(labels), random_state=42)
    print(f"Размер датасета: {len(all_images)} изображений")

    # Кодирование меток
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    unique_classes = len(np.unique(encoded_labels))
    print(f"Количество уникальных классов: {unique_classes}")

    # Нормализация изображений
    all_images = all_images / 255.0

    # Вывод по 5 изображений каждого класса перед аугментацией
    print("Отображение примеров изображений перед аугментацией...")
    plot_sample_images(all_images, encoded_labels, label_encoder, num_samples_per_class=5)

    print(f"Минимальное значение пикселя: {np.min(all_images)}, Максимальное значение пикселя: {np.max(all_images)}")

    # Аугментация данных
    print("\n=== Увеличение датасета через аугментацию ===")
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),  # Только горизонтальное отражение
        tf.keras.layers.RandomRotation(0.1),       # Поворот максимум на 10 градусов
    ])

    augmented_images = []
    augmented_labels = []
    num_augmentations_per_image = 10  # Количество аугментированных изображений на одно исходное

    for i, image in enumerate(all_images):
        augmented_images_list = augment_image(image, data_augmentation, num_augmentations_per_image)
        augmented_images.extend(augmented_images_list)
        augmented_labels.extend([encoded_labels[i]] * num_augmentations_per_image)
        print(f"Аугментировано изображение {i + 1}/{len(all_images)} (создано {num_augmentations_per_image} новых)")

    # Объединение исходных и аугментированных данных
    all_images = np.concatenate([all_images, np.array(augmented_images)], axis=0)
    all_labels = np.concatenate([encoded_labels, np.array(augmented_labels)], axis=0)
    print(f"\nИтоговый размер датасета: {len(all_images)} изображений")

    # Перемешивание данных после аугментации
    all_images, all_labels = shuffle(all_images, all_labels, random_state=42)
    print(f"Минимальное значение пикселя: {np.min(all_images)}, Максимальное значение пикселя: {np.max(all_images)}")
    # Ограничение значений пикселей в диапазоне [0, 1] (на всякий случай)
    all_images = np.clip(all_images, 0, 1)

    print(f"Минимальное значение пикселя: {np.min(all_images)}, Максимальное значение пикселя: {np.max(all_images)}")
    # Вывод по 5 изображений каждого класса после аугментации
    print("Отображение примеров изображений после аугментации...")
    plot_sample_images(all_images, all_labels, label_encoder, num_samples_per_class=5)


    # Разделение данных на тренировочную и валидационную выборки
    split_index = int(len(all_images) * 0.8)
    train_images, val_images = all_images[:split_index], all_images[split_index:]
    train_labels, val_labels = all_labels[:split_index], all_labels[split_index:]


        # Подсчет примеров каждого класса
    train_counts = np.bincount(train_labels)
    val_counts   = np.bincount(val_labels)
    classes  = label_encoder.classes_

    for cls, tc, vc in zip(classes, train_counts, val_counts):
        print(f"Класс «{cls}»: train = {tc}, val = {vc}")
    # Создание TensorFlow Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1000).batch(16)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(16)

    print("Форма train_images:", train_images.shape)
    print("Форма train_labels:", train_labels.shape)
    print("Форма val_images:  ", val_images.shape)
    print("Форма val_labels:  ", val_labels.shape)
    print(f"Диапазон пикселей train: [{train_images.min():.3f}, {train_images.max():.3f}]")
    print(f"Диапазон пикселей val:   [{val_images.min():.3f}, {val_images.max():.3f}]")

    # Проверка лейблов в первом батче
    for imgs, lbls in train_dataset.take(1):
        unique, counts = np.unique(lbls.numpy(), return_counts=True)
        print("labels in batch:", dict(zip(unique, counts)))

    # Определение коллбеков
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10, 
        restore_best_weights=True,
        min_delta=1e-4  
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    model = Sequential([
        Conv2D(32, 3, activation='relu', input_shape=(128,128,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),  
        
        Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.3),
        
        Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.4),
        
        Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.5),
        
        Flatten(),
        
        Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(unique_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Компиляция модели
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,  
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    print("Обучение завершено. Сохранение модели...")

    model_dir = 'flask_app/model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(os.path.join(model_dir, 'improved_card_predictor.keras'))
    np.save(os.path.join(model_dir, 'label_encoder.npy'), label_encoder.classes_)
    print("Модель и кодировщик меток успешно сохранены")

        # Вставка информации в таблицу DATASET_INFO
    conn = get_db_connection()
    cursor = conn.cursor()
    insert_query = (
        "INSERT INTO DATASET_INFO (DATASET_NAME, TOTAL_PHOTOS, TOTAL_CLIENTS) "
        "VALUES (?, ?, ?)"
    )
    # Подставьте нужное имя датасета в 'CLIENT_BIOMETRY'
    cursor.execute(insert_query, (
        'FACE_ID',  # DATASET_NAME
        len(all_images),     # TOTAL_PHOTOS
        unique_classes,      # TOTAL_CLIENTS
    ))
    log_query(insert_query)
    conn.commit()
    cursor.close()
    conn.close()
    print("Информация о датасете вставлена в DATASET_INFO")

    print("Отображение предсказаний модели...")
    plot_predictions(model, val_images, val_labels, label_encoder, num_samples=9)


    predictions = model.predict(val_images)
    predicted_labels = np.argmax(predictions, axis=1)
    print(f"Распределение предсказаний: {np.bincount(predicted_labels)}")

if __name__ == '__main__':
    create_dataset_and_train_model()
