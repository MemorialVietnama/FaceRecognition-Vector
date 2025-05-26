import os
from flask import Flask, redirect, request
from routes import init_routes



app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Отключаем экранирование Unicode
# Установка секретного ключа для сессий
app.secret_key = 'c62c5f07567cabedf24f269b024cace411b3d7a47cf29235711313164b55b7f8'

    
# Инициализация маршрутов
init_routes(app)

if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=8080,
        debug=True,
        threaded=True  # Включаем многопоточность
    )