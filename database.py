# database.py
import fdb
from datetime import datetime
import sys

def log_query(query, params=None):
    """Логирование SQL-запросов"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Executing query:")
    print("SQL:", query)
    if params:
        print("Parameters:", params)
    else:
        print("No parameters")


import fdb
import os

import fdb
import os

def get_db_connection():
    """Устанавливает соединение с базой данных Firebird."""
    print("\n" + "="*50)
    print("Establishing database connection...")
    
    try:
        # Явно задаём рабочую директорию (если нужно)
        os.chdir(r"C:\Users\Vlad\Desktop\kursa4_2025")
        
        # Определяем относительный путь к базе данных
        relative_path = os.path.join(
            "is122-java-Ewing", 
            "src", 
            "main", 
            "resources", 
            "org", 
            "example", 
            "atm_maven_jfx", 
            "Database", 
            "ATM_MODEL_DBASE.fdb"
        )
        
        # Разрешаем относительный путь в абсолютный
        db_path = os.path.abspath(relative_path)
        
        # Проверяем, существует ли файл
        if not os.path.exists(db_path):
            print(f"Database file not found at: {db_path}")
            raise FileNotFoundError(f"Database file not found at: {db_path}")
        
        print(f"Using database file: {db_path}")
        
        # Устанавливаем соединение с базой данных
        con = fdb.connect(
            dsn=f'localhost/3050:{db_path}',
            user='SYSDBA',
            password='010802',
            charset='UTF8'
        )
        print("Connection established successfully")
        return con
    
    except fdb.fbcore.DatabaseError as e:
        print(f"Database error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


def get_clients():
    """Получает данные о клиентах с номерами карт и балансом."""
    print("\n" + "="*50)
    print("Fetching clients data...")
    con = get_db_connection()
    cur = con.cursor()
    
    query = """
        SELECT 
            ci.FULL_FIO,
            cc.NUMBER_CARD,
            bc.BALANCE
        FROM CLIENT_INFO ci
        JOIN CLIENT_CARD cc ON ci.FULL_FIO = cc.FK_CLIENT
        JOIN BALANCE_CARD bc ON cc.NUMBER_CARD = bc.FK_CARD
    """
    log_query(query)
    
    cur.execute(query)
    clients_data = cur.fetchall()
    
    print(f"Retrieved {len(clients_data)} clients")
    for i, client in enumerate(clients_data, 1):
        print(f"{i}. {client[0]} | Card: {client[1]} | Balance: {client[2]}")
    
    cur.close()
    con.close()
    return clients_data

def add_client_to_db(client_data):
    """Добавляет нового клиента в базу данных."""
    print("\n" + "="*50)
    print("Adding new client to database...")
    print("Client data:", client_data)
    
    con = None
    cur = None
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # Сначала получаем новый ID
        cur.execute("SELECT COALESCE(MAX(ID), 0) + 1 FROM CLIENT_INFO")
        new_id = cur.fetchone()[0]
        print(f"Generated new ID: {new_id}")
        
        query = """
            INSERT INTO CLIENT_INFO (
                ID, FULL_FIO, SURNAME, NAME_CLIENT, NAME_FATHER, 
                AGE, GENGER, DATA_BIRTH, PASSPORT, 
                WHERE_PASSPORT, DATE_PASSPORT, SNILS, INN, STATUS
            ) 
            VALUES (
                ?, ?, ?, ?, ?, 
                ?, ?, ?, ?, 
                ?, ?, ?, ?, ?
            )
        """
        params = (
            new_id,  # Используем сгенерированный ID
            client_data['full_fio'],
            client_data['surname'],
            client_data['name_client'],
            client_data['name_father'],
            client_data['age'],
            client_data['genger'],
            client_data['data_birth'],
            client_data['passport'],
            client_data['where_passport'],
            client_data['date_passport'],
            client_data['snils'],
            client_data['inn'],
            client_data['status']
        )
        
        log_query(query, params)
        cur.execute(query, params)
        con.commit()
        print("Client added successfully")
        return True
        
    except Exception as e:
        if con:
            con.rollback()
        print(f"Error adding client: {str(e)}")
        raise e
    finally:
        if cur:
            cur.close()
        if con:
            con.close()

def get_client_by_card(card_number):
    print("\n" + "="*50)
    print(f"Fetching client by card number: {card_number}")
    con = get_db_connection()
    cur = con.cursor()
    
    query = """
        SELECT 
            ci.FULL_FIO,
            cc.NUMBER_CARD
        FROM CLIENT_INFO ci
        JOIN CLIENT_CARD cc ON ci.FULL_FIO = cc.FK_CLIENT
        WHERE cc.NUMBER_CARD = ?
    """
    log_query(query, (card_number,))
    
    cur.execute(query, (card_number,))
    client = cur.fetchone()
    
    if client:
        print(f"Found client: {client[0]} | Card: {client[1]}")
    else:
        print("Client not found")
    
    cur.close()
    con.close()
    return client

def update_client(old_card_number, full_fio, new_card_number, balance):
    print("\n" + "="*50)
    print(f"Updating client data. Old card: {old_card_number}, New card: {new_card_number}")
    print(f"Full FIO: {full_fio}, Balance: {balance}")
    
    con = get_db_connection()
    cur = con.cursor()
    
    try:
        # Обновление CLIENT_INFO
        query1 = """
            UPDATE CLIENT_INFO 
            SET FULL_FIO = ?
            WHERE FULL_FIO = (SELECT FK_CLIENT FROM CLIENT_CARD WHERE NUMBER_CARD = ?)
        """
        log_query(query1, (full_fio, old_card_number))
        cur.execute(query1, (full_fio, old_card_number))
        
        # Обновление CLIENT_CARD
        query2 = """
            UPDATE CLIENT_CARD 
            SET FK_CLIENT = ?, NUMBER_CARD = ?
            WHERE NUMBER_CARD = ?
        """
        log_query(query2, (full_fio, new_card_number, old_card_number))
        cur.execute(query2, (full_fio, new_card_number, old_card_number))
        
        # Обновление BALANCE_CARD
        query3 = """
            UPDATE BALANCE_CARD 
            SET FK_CARD = ?, BALANCE = ?
            WHERE FK_CARD = ?
        """
        log_query(query3, (new_card_number, balance, old_card_number))
        cur.execute(query3, (new_card_number, balance, old_card_number))
        
        con.commit()
        print("Client data updated successfully")
        return True
    except Exception as e:
        con.rollback()
        print(f"Error updating client: {str(e)}")
        raise e
    finally:
        cur.close()
        con.close()

def delete_client_db(card_number):
    """Удаляет клиента по номеру карты."""
    con = get_db_connection()
    cur = con.cursor()
    
    try:
        # Сначала получаем FULL_FIO из CLIENT_CARD
        cur.execute("SELECT FK_CLIENT FROM CLIENT_CARD WHERE NUMBER_CARD = ?", (card_number,))
        client_fio = cur.fetchone()
        
        if client_fio:
            client_fio = client_fio[0]
            # Удаляем из BALANCE_CARD
            cur.execute("DELETE FROM BALANCE_CARD WHERE FK_CARD = ?", (card_number,))
            # Удаляем из CLIENT_CARD
            cur.execute("DELETE FROM CLIENT_CARD WHERE NUMBER_CARD = ?", (card_number,))
            # Удаляем из CLIENT_INFO
            cur.execute("DELETE FROM CLIENT_INFO WHERE FULL_FIO = ?", (client_fio,))
            
            con.commit()
            return True
        return False
        
    except Exception as e:
        con.rollback()
        raise e
    finally:
        cur.close()
        con.close()
def check_passport_unique(passport):
    print("\n" + "="*50)
    print(f"Checking passport uniqueness: {passport}")
    con = get_db_connection()
    cur = con.cursor()
    
    query = "SELECT COUNT(*) FROM CLIENT_INFO WHERE PASSPORT = ?"
    log_query(query, (passport,))
    
    cur.execute(query, (passport,))
    count = cur.fetchone()[0]
    
    print(f"Passport {'already exists' if count > 0 else 'is unique'}")
    
    cur.close()
    con.close()
    return count == 0
def check_snils_unique(snils):
    con = get_db_connection()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM CLIENT_INFO WHERE SNILS = ?", (snils,))
    count = cur.fetchone()[0]
    cur.close()
    con.close()
    return count == 0
def check_inn_unique(inn):
    con = get_db_connection()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM CLIENT_INFO WHERE INN = ?", (inn,))
    count = cur.fetchone()[0]
    cur.close()
    con.close()
    return count == 0
# database.py (дополнение)
def get_banks():
    con = get_db_connection()
    cur = con.cursor()
    cur.execute("SELECT SMALL_TITTLE FROM BANK_DIC")
    banks = cur.fetchall()
    cur.close()
    con.close()
    return banks
def get_card_types():
    """Получает список типов карт из таблицы DIC_CARD_TYPE"""
    con = get_db_connection()
    cur = con.cursor()
    cur.execute("SELECT ID, TYPE_CARD FROM DIC_CARD_TYPE")
    card_types = cur.fetchall()
    cur.close()
    con.close()
    return card_types
def check_card_number_unique(card_number):
    """Проверяет уникальность номера карты"""
    con = get_db_connection()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM CLIENT_CARD WHERE NUMBER_CARD = ?", (card_number,))
    count = cur.fetchone()[0]
    cur.close()
    con.close()
    return count == 0
def create_client_card(card_data):
    """Создает карточку клиента и баланс"""
    print("\n" + "="*50)
    print("Начало создания карточки клиента...")
    print("Полученные данные:", card_data)
    
    con = None
    cur = None
    try:
        con = get_db_connection()
        cur = con.cursor()
        
        # 1. Запрос для получения типа карты
        type_card_query = "SELECT TYPE_CARD FROM DIC_CARD_TYPE WHERE ID = ?"
        type_card_params = (card_data['card_type_id'],)
        
        print("\nВыполняем запрос на получение типа карты:")
        print("SQL:", type_card_query)
        print("Параметры:", type_card_params)
        
        cur.execute(type_card_query, type_card_params)
        card_type = cur.fetchone()
        
        if not card_type:
            raise ValueError("Тип карты с указанным ID не найден")
        
        print(f"Найден тип карты: {card_type[0]}")
        
        # 2. Запрос на вставку в CLIENT_CARD
        insert_card_query = """
            INSERT INTO CLIENT_CARD (
                ID, FK_CLIENT, NUMBER_CARD, PIN_CODE, 
                FK_CARD_BANK, VALIDATION, FK_TYPE_CARD
            ) 
            VALUES (
                GEN_ID(GEN_CLIENT_CARD_ID, 1), ?, ?, ?, 
                ?, ?, ?
            )
        """
        insert_card_params = (
            card_data['client_fio'],
            card_data['card_number'],
            card_data['pin_code'],
            card_data['bank_id'],
            card_data['expiration_date'],
            card_type[0]  # Используем TYPE_CARD вместо ID
        )
        
        print("\nВыполняем запрос на вставку карты:")
        print("SQL:", insert_card_query)
        print("Параметры:", insert_card_params)
        
        cur.execute(insert_card_query, insert_card_params)
        
        # 3. Запрос на вставку в BALANCE_CARD
        insert_balance_query = """
            INSERT INTO BALANCE_CARD (
                ID, FK_CARD, BALANCE
            ) 
            VALUES (
                GEN_ID(GEN_BALANCE_CARD_ID, 1), ?, ?
            )
        """
        insert_balance_params = (
            card_data['card_number'],
            card_data['balance']
        )
        
        print("\nВыполняем запрос на вставку баланса:")
        print("SQL:", insert_balance_query)
        print("Параметры:", insert_balance_params)
        
        cur.execute(insert_balance_query, insert_balance_params)
        
        con.commit()
        print("\nТранзакция успешно завершена")
        return True
        
    except Exception as e:
        if con:
            con.rollback()
        print(f"\nОшибка при создании карточки: {str(e)}")
        raise e
    finally:
        if cur:
            cur.close()
        if con:
            con.close()
        print("="*50 + "\n")
def get_biometry(card_number: str) -> list:
    """Get all biometry data for a card"""
    con = get_db_connection()
    cur = con.cursor()
    try:
        query = "SELECT PHOTO_INDEX, IMAGE_DATA FROM CLIENT_BIOMETRY WHERE CARD_NUMBER = ? ORDER BY PHOTO_INDEX"
        cur.execute(query, (card_number,))
        return cur.fetchall()
    except Exception as e:
        print(f"Error getting biometry: {str(e)}")
        raise
    finally:
        cur.close()
        con.close()
 # Новая функция для получения полной информации о карте
def get_full_card_info(card_number):
        con = get_db_connection()
        cur = con.cursor()
        
        try:
            query = """
            SELECT 
                ci.*, 
                cc.*, 
                bc.BALANCE,
                ct.TYPE_CARD as card_type_name,
                cp.SMALL_TITLE as bank_name
            FROM CLIENT_INFO ci
            JOIN CLIENT_CARD cc ON ci.FULL_FIO = cc.FK_CLIENT
            JOIN BALANCE_CARD bc ON cc.NUMBER_CARD = bc.FK_CARD
            JOIN DIC_CARD_TYPE ct ON cc.FK_TYPE_CARD = ct.TYPE_CARD
            JOIN COMPANY cp ON cc.FK_CARD_BANK = cp.SMALL_TITLE
            WHERE cc.NUMBER_CARD = ?
            """
            cur.execute(query, (card_number,))
            columns = [column[0] for column in cur.description]
            card_data = dict(zip(columns, cur.fetchone()))
            
            return card_data
        except Exception as e:
            print(f"Error getting card info: {str(e)}")
            return None
        finally:
            cur.close()
            con.close()

def get_full_client_data(card_number):
    """Получает полные данные клиента из всех таблиц"""
    con = get_db_connection()
    cur = con.cursor()
    
    try:
        query = """
        SELECT 
            ci.FULL_FIO, ci.PASSPORT, ci.SNILS, ci.INN,
            cc.NUMBER_CARD, cc.PIN_CODE,
            bc.BALANCE
        FROM CLIENT_INFO ci
        JOIN CLIENT_CARD cc ON ci.FULL_FIO = cc.FK_CLIENT
        JOIN BALANCE_CARD bc ON cc.NUMBER_CARD = bc.FK_CARD
        WHERE cc.NUMBER_CARD = ?
        """
        cur.execute(query, (card_number,))
        row = cur.fetchone()
        
        if not row:
            return None
            
        columns = [column[0] for column in cur.description]
        return dict(zip(columns, row))
        
    except Exception as e:
        print(f"Error getting client data: {str(e)}")
        return None
    finally:
        cur.close()
        con.close()

def update_client_full(original_card_number, new_card_number, full_fio, passport, snils, inn, pin_code, balance):
    """Обновляет данные клиента во всех связанных таблицах"""
    con = get_db_connection()
    cur = con.cursor()
    
    try:
        # Начинаем транзакцию
        con.begin()
        
        # 1. Обновляем CLIENT_INFO
        cur.execute("""
            UPDATE CLIENT_INFO 
            SET FULL_FIO = ?, PASSPORT = ?, SNILS = ?, INN = ?
            WHERE FULL_FIO = (SELECT FK_CLIENT FROM CLIENT_CARD WHERE NUMBER_CARD = ?)
        """, (full_fio, passport, snils, inn, original_card_number))
        
        # 2. Обновляем CLIENT_CARD
        cur.execute("""
            UPDATE CLIENT_CARD 
            SET NUMBER_CARD = ?, PIN_CODE = ?, FK_CLIENT = ?
            WHERE NUMBER_CARD = ?
        """, (new_card_number, pin_code, full_fio, original_card_number))
        
        # 3. Обновляем BALANCE_CARD
        cur.execute("""
            UPDATE BALANCE_CARD 
            SET FK_CARD = ?, BALANCE = ?
            WHERE FK_CARD = ?
        """, (new_card_number, balance, original_card_number))
        
        # Фиксируем изменения
        con.commit()
        
    except Exception as e:
        con.rollback()
        raise e
    finally:
        cur.close()
        con.close()

# Добавить в database.py

def get_dataset_stats():
    """Получает статистику по датасетам"""
    con = get_db_connection()
    cur = con.cursor()
    try:
        # Получаем общее количество фотографий из CLIENT_BIOMETRY
        cur.execute("SELECT COUNT(*) FROM CLIENT_BIOMETRY")
        total_photos = cur.fetchone()[0]
        
        # Получаем дату последнего добавления фото
        cur.execute("SELECT MAX(CREATED_AT) FROM CLIENT_BIOMETRY")
        last_augmentation = cur.fetchone()[0]

        # Получаем информацию о последнем датасете
        
        return {
            'total_photos': total_photos,
            'estimated_augmented': total_photos * 5,  # Примерная оценка
            'last_augmentation': last_augmentation,
        }
    except Exception as e:
        print(f"Error getting dataset stats: {str(e)}")
        return None
    finally:
        cur.close()
        con.close()

def save_client_biometry(card_number: str, photo_index: int, image_data: bytes) -> bool:
    """
    Сохраняет биометрические данные клиента в базу данных.
    
    :param card_number: Номер карты клиента
    :param photo_index: Индекс изображения для данного клиента
    :param image_data: Бинарные данные изображения
    :return: True, если сохранение прошло успешно, иначе False
    """
    con = get_db_connection()
    cur = con.cursor()
    try:
        query = """
            INSERT INTO CLIENT_BIOMETRY 
            (CARD_NUMBER, PHOTO_INDEX, IMAGE_DATA)
            VALUES (?, ?, ?)
        """
        params = (card_number, photo_index, image_data)
        
        log_query(query, params)  # Логирование запроса (если необходимо)
        cur.execute(query, params)
        con.commit()
        return True
    except Exception as e:
        con.rollback()
        print(f"Ошибка при сохранении биометрии: {str(e)}", file=sys.stderr)
        return False
    finally:
        cur.close()
        con.close()

def get_client_biometry(card_number: str) -> list:
    """
    Получает все биометрические данные для указанного номера карты.
    
    :param card_number: Номер карты клиента
    :return: Список кортежей с данными (PHOTO_INDEX, IMAGE_DATA)
    """
    con = get_db_connection()
    cur = con.cursor()
    try:
        query = """
            SELECT PHOTO_INDEX, IMAGE_DATA
            FROM CLIENT_BIOMETRY 
            WHERE CARD_NUMBER = ?
            ORDER BY PHOTO_INDEX
        """
        log_query(query, (card_number,))  # Логирование запроса (если необходимо)
        cur.execute(query, (card_number,))
        return cur.fetchall()
    except Exception as e:
        print(f"Ошибка при получении биометрии: {str(e)}", file=sys.stderr)
        raise
    finally:
        cur.close()
        con.close()
def clear_client_biometry(card_number: str) -> bool:
    """Удаляет все биометрические данные для указанного номера карты"""
    con = get_db_connection()
    cur = con.cursor()
    try:
        query = "DELETE FROM CLIENT_BIOMETRY WHERE CARD_NUMBER = ?"
        log_query(query, (card_number,))
        cur.execute(query, (card_number,))
        con.commit()
        return cur.rowcount > 0
    except Exception as e:
        con.rollback()
        print(f"Ошибка при удалении биометрии: {str(e)}")
        raise
    finally:
        cur.close()
        con.close()
        
def get_biometry_stats() -> dict:
    """Возвращает статистику по биометрическим данным"""
    con = get_db_connection()
    cur = con.cursor()
    try:
        stats = {}
        
        # Общее количество фотографий
        cur.execute("SELECT COUNT(*) FROM CLIENT_BIOMETRY")
        stats['total_photos'] = cur.fetchone()[0]
        
        # Количество оригинальных фотографий (не аугментированных)
        cur.execute("SELECT COUNT(*) FROM CLIENT_BIOMETRY")
        stats['original_photos'] = cur.fetchone()[0]
        
        # Количество уникальных клиентов с фотографиями
        cur.execute("SELECT COUNT(DISTINCT CARD_NUMBER) FROM CLIENT_BIOMETRY")
        stats['clients_with_photos'] = cur.fetchone()[0]
        
        # Дата последнего добавления фото
        cur.execute("SELECT MAX(CREATED_AT) FROM CLIENT_BIOMETRY")
        stats['last_photo_date'] = cur.fetchone()[0]
        
        return stats
    except Exception as e:
        print(f"Ошибка при получении статистики: {str(e)}")
        raise
    finally:
        cur.close()
        con.close()

def get_dataset_summary():
    """
    Получает сводку по датасету: общее количество фото, аугментированных фото, уникальных клиентов.
    """
    con = get_db_connection()
    cur = con.cursor()
    try:
        summary = {}

        # Общее количество оригинальных фотографий
        query = "SELECT COUNT(*) FROM CLIENT_BIOMETRY"
        log_query(query)
        cur.execute(query)
        summary['total_original_photos'] = cur.fetchone()[0]

        # Оценка аугментированных фотографий (10 аугментаций на каждое оригинальное фото)
        summary['estimated_augmented'] = summary['total_original_photos'] * 10

        # Количество уникальных клиентов
        query = "SELECT COUNT(DISTINCT CARD_NUMBER) FROM CLIENT_BIOMETRY"
        log_query(query)
        cur.execute(query)
        summary['unique_clients'] = cur.fetchone()[0]

        # Общее количество аугментированных фотографий на основе предыдущих сборок (если хранится в базе)
        summary['total_augmented_photos'] = summary['total_original_photos'] + summary['estimated_augmented']

        # Дата последней аугментации
        query = "SELECT MAX(CREATED_AT) FROM CLIENT_BIOMETRY"
        log_query(query)
        cur.execute(query)
        summary['last_augmentation'] = cur.fetchone()[0]

        return summary
    except Exception as e:
        print(f"Ошибка при получении сводки датасета: {str(e)}")
        raise
    finally:
        cur.close()
        con.close()


