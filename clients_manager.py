import fdb

def get_db_connection():
    """Устанавливает соединение с базой данных Firebird."""
    con = fdb.connect(
        dsn='localhost/3050:/home/vladislav/IdeaProjects/is122-java-Ewing/src/main/resources/org/example/atm_maven_jfx/Database/ATM_MODEL_DBASE.fdb',
        user='SYSDBA',
        password='010802',
        charset='UTF8'
    )
    return con

def get_clients():
    """Получает данные о клиентах с номерами карт и балансом."""
    con = get_db_connection()
    cur = con.cursor()
    
    # Запрос с JOIN для получения ФИО, номера карты и баланса
    cur.execute("""
        SELECT 
            ci.FULL_FIO,
            cc.NUMBER_CARD,
            bc.BALANCE
        FROM CLIENT_INFO ci
        JOIN CLIENT_CARD cc ON ci.FULL_FIO = cc.FK_CLIENT
        JOIN BALANCE_CARD bc ON cc.NUMBER_CARD = bc.FK_CARD
    """)
    clients_data = cur.fetchall()
    
    cur.close()
    con.close()
    
    return clients_data