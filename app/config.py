import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    # server-side session backend
    SESSION_TYPE = "filesystem"
    SECRET_KEY = "349874329587348923798471239"
    # database
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Uploads
    ALLOWED_EXTENSIONS = set(["txt", "csv"])
    ALLOWED_FILE_SIZE = 26214400  # (in bytes) = 25 MB
    # folders
    if False:  # os.environ["PWD"] == "/":  # web server
        SESSION_FILE_DIR = "/var/www/PATTIE_User_Modeling/flask_session"
        DATA_DIR = "/var/www/PATTIE_User_Modeling/data"
        UPLOAD_FOLDER = "/var/www/saved_files/uploaded"  # needs to be changed
        SQLALCHEMY_DATABASE_URI = os.environ.get(
            "DATABASE_URL"
        ) or "sqlite:///" + os.path.join("/var/www/saved_files/", "pattie.db")
    else:  # local
        SESSION_FILE_DIR = "flask_session"
        DATA_DIR = "data"
        UPLOAD_FOLDER = "uploaded"
        SQLALCHEMY_DATABASE_URI = os.environ.get(
            "DATABASE_URL"
        ) or "sqlite:///" + os.path.join(basedir, "app.db")

    # DC_DB_USER = 'postgres'
    # DC_DB_PASS = 'Arvi1308'
    # DC_DB_HOST = 'localhost'
    # DC_DB_PORT = 5433
    # DC_DB_NAME = 'PATTIE'
    
    DC_DB_USER = 'aravind'
    DC_DB_PASS = 'C&99Fk6xHxypA2R$C4XQ'
    DC_DB_HOST = '34.133.177.246'
    DC_DB_PORT = 5432
    DC_DB_NAME = 'dcdi'
    

    DB_CONFIG = {
        "user": DC_DB_USER,
        "password": DC_DB_PASS,
        "host": DC_DB_HOST,
        "port": DC_DB_PORT,
        "dbname": DC_DB_NAME,
    }
    # ELASTICSEARCH_HOST = os.environ["ELASTICSEARCH_HOST"]
