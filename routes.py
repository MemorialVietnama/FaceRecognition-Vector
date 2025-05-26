import base64
import json
import sys
from flask import render_template, redirect, url_for, request, flash, jsonify, session, make_response
from datetime import datetime, timedelta
import time  # Добавьте этот импорт
from io import StringIO
from database import (
    get_db_connection, get_clients, add_client_to_db,
    delete_client_db, check_passport_unique, check_snils_unique,
    check_inn_unique, get_banks, get_card_types, check_card_number_unique,
    create_client_card,get_full_client_data, update_client_full,
    get_client_biometry, save_client_biometry,get_client_by_card
)
from flask import stream_with_context, Response

import tempfile
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import io
from train_vector_model import *
from test_vector_new import *
from test_vector import *
import logging
import joblib
import werkzeug

# Глобальная переменная для камеры
camera = None
# Глобальная переменная для хранения состояния
training_in_progress = False

def allowed_file(filename):
    """Проверяем, что файл имеет разрешенное расширение"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}
def get_full_name_from_card(card_number):
    """Получение полного имени клиента по номеру карты"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Исправленный SQL-запрос с JOIN и правильным синтаксисом параметров
        query = """
            SELECT FK_CLIENT FROM CLIENT_CARD WHERE NUMBER_CARD = ?
        """
        
        cursor.execute(query, (card_number,))
        result = cursor.fetchone()
        
        return result[0] if result else "Неизвестный пользователь"
    
    except Exception as e:
        logging.error(f"Database error: {str(e)}")
        return "Ошибка получения данных"
    
    finally:
        if conn:
            conn.close()
def generate_training_events():
    global training_in_progress
    training_in_progress = True

    def send_event(message):
        yield f"data: {message}\n\n"

    result = process_and_train(send_event)
    training_in_progress = False
    if result:
        yield "data: 🎯 Обучение завершено успешно!\n\n"
    else:
        yield "data: 🚨 Обучение завершено с ошибками!\n\n"
def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        # Устанавливаем разрешение
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera
# Глобальная инициализация тестера с обработкой ошибок


def generate_frames():
    """Генератор кадров с улучшенной обработкой ошибок"""
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        logging.error("Camera initialization failed")
        return
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    try:
        while True:
            success, frame = camera.read()
            if not success:
                logging.warning("Frame capture failed")
                break

            # Обработка кадра
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Кодирование кадра
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    except Exception as e:
        logging.error(f"Video stream error: {str(e)}")
    finally:
        camera.release()
        logging.info("Camera resources released")
# Константы
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_DIR = SCRIPT_DIR / 'model'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Глобальные переменные для моделей
encoder = tf.keras.models.load_model(MODEL_DIR / 'encoder.keras')

with open(MODEL_DIR / 'pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open(MODEL_DIR / 'class_params.pca.pkl', 'rb') as f:
    class_params = pickle.load(f)
    vectors_2d = class_params['vectors_2d']
    card_numbers = class_params['card_numbers']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_face(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple = (30, 30),
    target_size: tuple = (224, 224),
) -> np.ndarray | None:
    if not hasattr(detect_face, "face_cascade"):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        detect_face.face_cascade = cv2.CascadeClassifier(cascade_path)
        if detect_face.face_cascade.empty():
            raise RuntimeError("Не удалось загрузить каскад Хаара")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect_face.face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_image = image[y:y+h, x:x+w]
    return cv2.resize(face_image, target_size, interpolation=cv2.INTER_AREA)
        
def preprocess_image_array(image: np.ndarray) -> np.ndarray | None:
    # Конвертация в RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Обнаружение лица
    face_image = detect_face(image)
    if face_image is None:
        return None

    # Предобработка под модель
    face_image = preprocess_input(face_image.astype('float32'))
    return np.expand_dims(face_image, axis=0)  # Добавление batch dimension
def init_routes(app):
    @app.before_request
    def before_request():
        # Инициализация сессии для каждого запроса
        session.permanent = True
        app.permanent_session_lifetime = timedelta(minutes=30)
        
    @app.route('/webcam')
    def webcam():
        return render_template('webcam.html')

    @app.route('/recognize', methods=['POST'])
    def recognize_face():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            try:
                # Чтение файла как массива
                file_bytes = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    return jsonify({'error': 'Failed to decode image'}), 400

                # Предобработка
                processed_image = preprocess_image_array(image)
                if processed_image is None:
                    return jsonify({'error': 'Face detection failed'}), 400

                # Получение эмбеддинга
                embedding = encoder.predict(processed_image, verbose=0)
                new_point = pca.transform(embedding)
                distances = np.linalg.norm(vectors_2d - new_point, axis=1)
                sorted_indices = np.argsort(distances)

                results = []
                for idx in sorted_indices[:3]:
                    distance = float(distances[idx])
                    card_number = card_numbers[idx]

                    if distance < 80:
                        status = "CONFIDENT_MATCH"
                        confidence = max(0, 1 - distance / 100)
                    elif distance < 100:
                        status = "PARTIAL_MATCH"
                        confidence = max(0, 0.7 - (distance - 80) / 50)
                    else:
                        status = "NO_MATCH"
                        confidence = 0.0

                    results.append({
                        'card_number': card_number,
                        'distance': distance,
                        'status': status,
                        'confidence': round(confidence, 2)
                    })

                # Обработка результатов после цикла
                best_match = results[0] if results else None

                if not best_match or best_match['status'] == "NO_MATCH":
                    return jsonify({
                        'status': 'no_match',
                        'recognition_result': {
                            'status': 'no_match',
                            'primary_match': None,
                            'alternative_matches': [
                                m for m in results 
                                if m['status'] != "NO_MATCH"
                            ]
                        }
                    }), 200  # Используем 200 OK даже для no_match

                # Получение данных из БД с обработкой ошибок
                try:
                    full_name = get_full_name_from_card(best_match['card_number'])
                except Exception as e:
                    app.logger.error(f"Database error: {str(e)}")
                    full_name = "Неизвестный пользователь"

                response = {
                    'status': 'success',
                    'recognition_result': {
                        'status': best_match['status'].lower(),
                        'primary_match': {
                            'card_number': best_match['card_number'],
                            'client_info': {
                                'full_name': full_name,
                                'card_id': best_match['card_number']
                            },
                            'confidence': best_match['confidence']
                        },
                        'alternative_matches': [
                            {
                                'card_number': m['card_number'],
                                'confidence': m['confidence']
                            } for m in results[1:] 
                            if m['status'] != "NO_MATCH"
                        ]
                    }
                }

                return jsonify(response)

            except Exception as e:
                app.logger.error(f"Recognition error: {str(e)}", exc_info=True)
                return jsonify({
                    'status': 'error',
                    'message': 'Internal server error'
                }), 500

        return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400

                
    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

        # Новый маршрут для manage_model.html
    @app.route('/manage_model')
    def manage_model():
        return render_template('manage_model.html')

    @app.route('/home')
    def home():
        return render_template('index.html')

    @app.route('/clients')
    def clients():
        clients_data = get_clients()
        return render_template('clients.html', clients=clients_data)

    @app.route('/add_client', methods=['GET', 'POST'])
    def add_client():
        if request.method == 'POST':
            try:
                # Преобразование дат из формы
                data_birth = datetime.strptime(request.form['data_birth'], '%d.%m.%Y').date()
                date_passport = datetime.strptime(request.form['date_passport'], '%d.%m.%Y').date()
            except ValueError:
                flash('Неверный формат даты. Используйте ДД.ММ.ГГГГ', 'error')
                return redirect(url_for('add_client'))

            client_data = {
                'full_fio': request.form['full_fio'],
                'surname': request.form['surname'],
                'name_client': request.form['name_client'],
                'name_father': request.form['name_father'],
                'age': request.form['age'],
                'genger': request.form['genger'],
                'data_birth': data_birth,
                'passport': request.form['passport'],
                'where_passport': request.form['where_passport'],
                'date_passport': date_passport,
                'snils': request.form['snils'],
                'inn': request.form['inn'],
                'status': request.form['status']
            }

            # Проверка уникальности данных
            if not check_passport_unique(client_data['passport']):
                flash('Паспорт с таким номером уже существует', 'error')
                return redirect(url_for('add_client'))

            if not check_snils_unique(client_data['snils']):
                flash('СНИЛС с таким номером уже существует', 'error')
                return redirect(url_for('add_client'))

            if not check_inn_unique(client_data['inn']):
                flash('ИНН с таким номером уже существует', 'error')
                return redirect(url_for('add_client'))

            try:
                add_client_to_db(client_data)
                return redirect(url_for('create_card', client_fio=client_data['full_fio']))
            except Exception as e:
                flash(f'Ошибка при добавлении клиента: {str(e)}', 'error')
                return redirect(url_for('add_client'))

        return render_template('add_client.html')

    @app.route('/create_card', methods=['GET', 'POST'])
    def create_card():
        if request.method == 'POST':
            expiration_date = (datetime.now() + timedelta(days=5*365)).strftime('%m/%y')
            
            card_data = {
                'client_fio': request.form['client_fio'],
                'card_number': request.form['card_number'],
                'pin_code': request.form['pin_code'],
                'bank_id': request.form['bank_id'],
                'expiration_date': expiration_date,
                'card_type_id': request.form['card_type_id'],
                'balance': float(request.form['balance']) if request.form['balance'] else 0.0
            }

            if not check_card_number_unique(card_data['card_number']):
                flash('Карта с таким номером уже существует', 'error')
                return redirect(url_for('create_card', client_fio=card_data['client_fio']))

            try:
                create_client_card(card_data)
                
                # Получаем данные созданной карты
                card_info = {
                    'FULL_FIO': card_data['client_fio'],
                    'NUMBER_CARD': card_data['card_number'],
                    'BALANCE': card_data['balance']
                }
                
                return render_template('card_creation_success.html', card_info=card_info)
                
            except Exception as e:
                flash(f'Ошибка при создании карты: {str(e)}', 'error')
                return redirect(url_for('create_card', client_fio=card_data['client_fio']))

        client_fio = request.args.get('client_fio')
        banks = get_banks()
        card_types = get_card_types()
        return render_template('create_card.html', 
                            client_fio=client_fio,
                            banks=banks,
                            card_types=card_types)

    @app.route('/card_creation_success')
    def card_creation_success():
        if 'last_created_card' not in session:
            return redirect(url_for('clients'))
        
        card_info = session['last_created_card']
        return render_template('card_creation_success.html', card_info=card_info)


    @app.route('/check_card_number')
    def check_card_unique():
        card_number = request.args.get('number')
        is_unique = check_card_number_unique(card_number)
        return jsonify({'is_unique': is_unique})

    @app.route('/edit_client/<card_number>', methods=['GET', 'POST'])
    def edit_client(card_number):
        if request.method == 'POST':
            try:
                # Получаем все данные из формы
                full_fio = request.form['full_fio']
                new_card_number = request.form['number_card']
                balance = float(request.form['balance']) if request.form['balance'] else 0.0
                passport = request.form.get('passport', '')
                snils = request.form.get('snils', '')
                inn = request.form.get('inn', '')
                pin_code = request.form.get('pin_code', '')

                # Обновляем данные во всех таблицах
                update_client_full(
                    original_card_number=card_number,
                    new_card_number=new_card_number,
                    full_fio=full_fio,
                    passport=passport,
                    snils=snils,
                    inn=inn,
                    pin_code=pin_code,
                    balance=balance
                )

                flash('Данные клиента успешно обновлены', 'success')
                return redirect(url_for('clients'))
                
            except Exception as e:
                flash(f'Ошибка при обновлении данных: {str(e)}', 'error')
                return redirect(url_for('edit_client', card_number=card_number))

        # Для GET запроса - получаем полные данные клиента
        client_data = get_full_client_data(card_number)
        if not client_data:
            flash('Клиент не найден', 'error')
            return redirect(url_for('clients'))
        
        return render_template('edit_client.html', client=client_data)


    @app.route('/delete_client/<card_number>', methods=['POST'])
    def delete_client(card_number):
        try:
            delete_client_db(card_number)
            flash('Клиент успешно удален', 'success')
            return redirect(url_for('clients'))
        except Exception as e:
            flash(f'Ошибка при удалении клиента: {str(e)}', 'error')
            return redirect(url_for('clients'))
        
    @app.route('/biometry/<card_number>', methods=['GET', 'POST'])
    def biometry(card_number):
        if request.method == 'POST':
            try:
                # Получаем текущее количество фото клиента
                existing_photos = get_client_biometry(card_number)
                photo_index = len(existing_photos) + 1
                
                # Сохраняем новое фото
                image_data = request.json['image_data']
                save_client_biometry(card_number, photo_index, image_data)
                
                return jsonify({
                    'success': True,
                    'photo_count': photo_index
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        # GET запрос - отображаем страницу
        photo_count = len(get_client_biometry(card_number))
        return render_template('biometry.html', 
                            card_number=card_number,
                            photo_count=photo_count)
    

    @app.route('/dataset')
    def dataset_management():
        try:
            con = get_db_connection()
            cur = con.cursor()

            # Получаем список клиентов с количеством фотографий
            cur.execute("""
                SELECT 
                    ci.FULL_FIO,
                    cc.NUMBER_CARD,
                    (SELECT COUNT(*) 
                    FROM CLIENT_BIOMETRY cb 
                    WHERE cb.CARD_NUMBER = cc.NUMBER_CARD) AS photos_count
                FROM CLIENT_INFO ci
                JOIN CLIENT_CARD cc ON ci.FULL_FIO = cc.FK_CLIENT
                ORDER BY ci.FULL_FIO;
            """)
            clients = cur.fetchall()

            # Общая статистика фотографий
            cur.execute("""
                SELECT COUNT(*) AS total_photos, MAX(CREATED_AT) AS last_augmentation 
                FROM CLIENT_BIOMETRY;
            """)
            stats = cur.fetchone()

            total_photos = stats[0] if stats else 0
            last_augmentation = stats[1] if stats else None

            # Количество уникальных клиентов с фотографиями
            cur.execute("SELECT COUNT(DISTINCT CARD_NUMBER) FROM CLIENT_BIOMETRY")
            clients_with_photos = cur.fetchone()[0]

            # Дата последней сборки датасета
            last_dataset_creation = None
            try:
                cur.execute("SELECT MAX(CREATED_AT) FROM DATASET_INFO")
                last_dataset_creation = cur.fetchone()[0]
            except:
                pass

            # Данные об аугментированных изображениях
            dataset_dir = os.path.join('dataset', 'images')
            unique_aug_clients = 0
            total_augmented_photos = 0

            if os.path.exists(dataset_dir):
                for folder in os.listdir(dataset_dir):
                    folder_path = os.path.join(dataset_dir, folder)
                    if os.path.isdir(folder_path):
                        unique_aug_clients += 1
                        total_augmented_photos += len([
                            f for f in os.listdir(folder_path)
                            if 'augmented' in f and os.path.isfile(os.path.join(folder_path, f))
                        ])

            estimated_augmented = total_photos * 11

            return render_template('dataset.html',
                                clients=clients,
                                total_photos=total_photos,
                                clients_with_photos=clients_with_photos,
                                last_augmentation=last_augmentation,
                                last_dataset_creation=last_dataset_creation,
                                unique_aug_clients=unique_aug_clients,
                                total_augmented_photos=total_augmented_photos,
                                estimated_augmented=estimated_augmented)

        except Exception as e:
            return render_template('error.html', error=str(e))
        finally:
            if cur:
                cur.close()
            if con:
                con.close()



    @app.route('/capture_photos/<card_number>', methods=['GET', 'POST'])
    def capture_photos(card_number):
        if request.method == 'POST':
            # Обработка загрузки фото
            pass
        return render_template('biometry.html', card_number=card_number)


    @app.route('/dataset/clear_photos/<card_number>', methods=['POST'])
    def clear_photos(card_number):
        try:
            con = get_db_connection()
            cur = con.cursor()
            cur.execute("DELETE FROM CLIENT_BIOMETRY WHERE CARD_NUMBER = ?", (card_number,))
            con.commit()
            flash(f'Фотографии для карты {card_number} успешно удалены', 'success')
        except Exception as e:
            con.rollback()
            flash(f'Ошибка при удалении фотографий: {str(e)}', 'error')
        finally:
            cur.close()
            con.close()
        return redirect(url_for('dataset_management'))

    @app.route('/api/validate_photos', methods=['POST'])
    def validate_photos():
        if 'photos' not in request.files:
            return jsonify({'success': False, 'error': 'No files uploaded'}), 400
        
        photo_files = request.files.getlist('photos')
        if not photo_files or photo_files[0].filename == '':
            return jsonify({'success': False, 'error': 'No files selected'}), 400
        

        temp_dir = tempfile.mkdtemp()
        temp_files = []
        results = {
            'success': True,
            'valid': [],
            'invalid': [],
            'all_same_person': False,
            'validation_passed': False
        }

        try:
            # Сохраняем файлы во временную директорию
            for i, photo in enumerate(photo_files):
                if photo.filename == '':
                    continue
                
                # Проверяем расширение файла
                ext = os.path.splitext(photo.filename)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png']:
                    results['invalid'].append({
                        'filename': photo.filename,
                        'error': 'Invalid file format'
                    })
                    continue
                
                temp_path = os.path.join(temp_dir, f"photo_{i}{ext}")
                photo.save(temp_path)
                temp_files.append(temp_path)

        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Server error: {str(e)}'
            }), 500
        
        finally:
            # Удаляем временные файлы
            for path in temp_files:
                try:
                    os.remove(path)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass

    @app.route('/api/save_photos', methods=['POST'])
    def save_photos():
        data = request.get_json()
        if not data or 'card_number' not in data or 'photos' not in data:
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400
        
        card_number = data['card_number']
        saved_count = 0
        
        try:
            con = get_db_connection()
            cur = con.cursor()
            
            # Получаем текущий индекс фото
            cur.execute("SELECT MAX(PHOTO_INDEX) FROM CLIENT_BIOMETRY WHERE CARD_NUMBER = ?", (card_number,))
            last_index = cur.fetchone()[0] or 0
            
            # Сохраняем фотографии
            for i, photo_data in enumerate(data['photos'], start=1):
                # В реальном приложении вам нужно будет обработать photo_data
                # Например, если вы сохраняете base64-encoded изображения:
                import base64
                img_data = base64.b64decode(photo_data['image'])
                
                cur.execute("""
                    INSERT INTO CLIENT_BIOMETRY 
                    (CARD_NUMBER, PHOTO_INDEX, IMAGE_DATA, CREATED_AT)
                    VALUES (?, ?, ?, ?)
                """, (card_number, last_index + i, img_data, datetime.now()))
                
                saved_count += 1
            
            con.commit()
            return jsonify({
                'success': True,
                'count': saved_count,
                'message': 'Photos saved successfully'
            })
        
        except Exception as e:
            if 'con' in locals():
                con.rollback()
            return jsonify({
                'success': False,
                'error': f'Database error: {str(e)}'
            }), 500
        
        finally:
            if 'cur' in locals():
                cur.close()
            if 'con' in locals():
                con.close()


    @app.route('/upload_photos', methods=['POST'])
    def upload_photos():
        try:
            # Проверка наличия файлов
            if 'photos' not in request.files:
                return make_response(
                    jsonify({'success': False, 'message': 'No files uploaded'}),
                    400,
                    {'Content-Type': 'application/json'}
                )

            card_number = request.form.get('card_number')
            if not card_number:
                return make_response(
                    jsonify({'success': False, 'message': 'Card number not provided'}),
                    400,
                    {'Content-Type': 'application/json'}
                )

            files = request.files.getlist('photos')
            if not files or files[0].filename == '':
                return make_response(
                    jsonify({'success': False, 'message': 'No files selected'}),
                    400,
                    {'Content-Type': 'application/json'}
                )

            con = get_db_connection()
            cur = con.cursor()
            
            cur.execute("SELECT MAX(PHOTO_INDEX) FROM CLIENT_BIOMETRY WHERE CARD_NUMBER = ?", (card_number,))
            last_index = cur.fetchone()[0] or 0
            
            saved_count = 0
            
            for i, file in enumerate(files, start=1):
                if file.filename == '':
                    continue
                    
                try:
                    # Проверка типа файла
                    if not (file.filename.lower().endswith(('.jpg', '.jpeg', '.png'))):
                        continue
                        
                    # Сохранение файла
                    img_bytes = file.read()
                    cur.execute("""
                        INSERT INTO CLIENT_BIOMETRY 
                        (CARD_NUMBER, PHOTO_INDEX, IMAGE_DATA, CREATED_AT)
                        VALUES (?, ?, ?, ?)
                    """, (card_number, last_index + i, img_bytes, datetime.now()))
                    saved_count += 1
                except Exception as e:
                    print(f"Error processing file {file.filename}: {str(e)}")
                    continue
            
            con.commit()
            
            response_data = {
                'success': True,
                'count': saved_count,
                'message': f'Successfully uploaded {saved_count} photos',
                'redirect': url_for('dataset_management')
            }
            
            return make_response(
                jsonify(response_data),
                200,
                {'Content-Type': 'application/json'}
            )
            
        except Exception as e:
            if 'con' in locals():
                con.rollback()
            return make_response(
                jsonify({'success': False, 'message': f'Server error: {str(e)}'}),
                500,
                {'Content-Type': 'application/json'}
            )
        finally:
            if 'cur' in locals():
                cur.close()
            if 'con' in locals():
                con.close()



    @app.route('/build_dataset_progress')
    def build_dataset_progress():
        try:
            print("Запуск процесса обучения...")
            process_and_train()  # Вызов функции
            return jsonify({"status": "success", "message": "Обучение завершено успешно!"})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Ошибка во время выполнения: {str(e)}"})
    




