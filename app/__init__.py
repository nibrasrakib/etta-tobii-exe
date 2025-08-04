import logging

from flask import Flask
from flask_cors import CORS
# TOBII 
from flask_socketio import SocketIO

from app.config import Config
from flask_session import Session

# setup logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config.from_object(Config)

# TOBII
CORS(app)  # Enable CORS
socketio = SocketIO(app, cors_allowed_origins="*",async_mode='threading')


# enable sessions
sess = Session()
sess.init_app(app)

from app import routes

'''
main
'''
if __name__ == '__main__':

    app.run('127.0.0.1', 5000, debug=True)