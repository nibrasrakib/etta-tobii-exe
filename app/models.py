from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from sqlalchemy.orm import relationship
from app import db, login_manager

# for authentication

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

class User(db.Model, UserMixin):
    __tablename__ = 'webgaze_users'  # Match the table name in PostgreSQL

    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)  # Store hashed passwords
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<User {self.username}>'

    def set_password(self, password):
        """Hashes the password and stores it."""
        self.password = generate_password_hash(password)

    def check_password(self, password):
        """Checks if the given password matches the stored hash."""
        return check_password_hash(self.password, password)
    
    def get_id(self):
        """Returns the user ID in string format."""
        return str(self.user_id)

    def to_json(self):
        """Converts user data to a dictionary format, excluding sensitive data."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email
        }
        
class WebGazeData(db.Model):
    __tablename__ = 'webgaze_data'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    gaze_data = db.Column(db.JSON, nullable=False)  # Store as JSONB in PostgreSQL

    def __repr__(self):
        return f'<WebGazeData {self.username}>'
    
    def to_json(self):
        """Converts user data to a dictionary format, excluding sensitive data."""
        return {
            'id': self.id,
            'username': self.username,
            'timestamp': self.timestamp,
            'gaze_data': self.gaze_data
        }