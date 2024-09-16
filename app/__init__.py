from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from app.config import Config
from flask_session import Session
from flask_login import LoginManager
import visual_library_plos as vl
import logging
import re

# setup logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config.from_object(Config)

# database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# authentication
login_manager = LoginManager(app)

# enable sessions
sess = Session()
sess.init_app(app)

# regular expressions
tokenize = re.compile("[^\w\-]+")  # a token is composed of
# alphanumerics or hyphens
exclude = re.compile("(\d+|.)$")  # number or one-character tokens

# Read stopword list
stopwords = vl.read_stopwords(app.config['DATA_DIR'])

# variables
MINDF = 1

from app import routes, routes_auth, routes_upload, utils, models

'''
main
'''
if __name__ == '__main__':

    app.run('127.0.0.1', 5000, debug=True)
    # app.run(processes=10, port=5000, debug=True)
