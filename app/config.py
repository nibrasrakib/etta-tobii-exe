import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    # server-side session backend
    print("Using server-side session backend")
    SESSION_TYPE = "filesystem"