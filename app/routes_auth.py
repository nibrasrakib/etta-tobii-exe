# for user login
from flask_login import current_user, login_user, logout_user
from flask import render_template, redirect, url_for, request, flash, json, current_app
from urllib.parse import quote_plus
from app import app, db
from app.forms import LoginForm, RegistrationForm
from app.models import User
import os

import uuid
from google.cloud import storage
from google.oauth2 import service_account


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     print("Called login route")  # Debug statement
#     target_url = request.args.get('target', '')
#     if current_user.is_authenticated:
#         return redirect(url_for('index'))

#     form = LoginForm()
#     if form.validate_on_submit():
#         print("Form validated successfully")  # Debug statement
#         user = User.query.filter_by(username=form.username.data).first()
#         if user is None:
#             print("No user found with that username")  # Debug statement
#         elif not user.check_password(form.password.data):
#             print("Password check failed")  # Debug statement

#         if user is None or not user.check_password(form.password.data):
#             flash('Invalid username or password', 'danger')
#             return redirect(url_for('login'))
        
#         login_user(user, remember=form.remember_me.data)
#         print("User logged in successfully")  # Debug statement
#         next_page = request.args.get('next')
#         return redirect(next_page) if next_page else redirect(url_for('index'))

#     print("Rendering login template")  # Debug statement
#     return render_template('login.html', form=form, encoded_target=request.args.get('target', ''))

@app.route('/login', methods=['GET', 'POST'])
def login():
    print("Called login route")
    
    # Capture the target URL from request arguments
    target_url = request.args.get("target", "")
    print("Target URL:", target_url)

    # If the user is already authenticated, redirect to the target URL
    if current_user.is_authenticated:
        return redirect(target_url or url_for("index"))  # Fallback to 'index' if no target_url provided

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        # Check if the user exists and the password is correct
        if user is None or not user.check_password(form.password.data):
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login") + "?target=" + quote_plus(target_url))
        
        # Log in the user
        login_user(user, remember=form.remember_me.data)
        flash("Logged in successfully!", "success")  # Flash success message
        print("User logged in successfully")

        # Redirect to target URL after login
        return redirect(target_url or url_for("index"))  # Fallback to 'index' if no target_url provided

    # Render login template with encoded target URL
    return render_template("login.html", form=form, encoded_target=quote_plus(target_url))

@app.route('/logout')
def logout():
    logout_user()
    target_url = request.args.get('target')
    return redirect(target_url)

# Works
@app.route('/register', methods=['GET', 'POST'])
def register():
    print("Called register")
    env = request.environ
    target_url = request.args.get('target')
    print(target_url)

    if 'eppn' in env:
        print('eppn in env')
        return redirect(target_url)
    if current_user.is_authenticated:
        print('current_user.is_authenticated')
        return redirect(target_url)

    form = RegistrationForm()
    if form.validate_on_submit():
        print('form validated')

        # Create a new user instance
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)  # Hash and set the password

        # Add and commit the new user to the database
        db.session.add(user)
        
        try:
            db.session.commit()  # Commit the transaction to save the user
            flash('Congratulations, you are now a registered user!')
            return redirect(url_for('login') + '?target=' + quote_plus(target_url))
        except Exception as e:
            db.session.rollback()  # Rollback if there is an error
            print(f"Error saving user to the database: {e}")
            flash('An error occurred while trying to register. Please try again.')

    print('rendering register.html')
    return render_template('register.html', 
                           encoded_target=quote_plus(target_url), 
                           title='Register', 
                           form=form)

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     print("Called register")
#     env = request.environ
#     target_url = request.args.get('target')
#     print(target_url)

#     if 'eppn' in env:
#         print('eppn in env')
#         return redirect(target_url)
#     if current_user.is_authenticated:
#         print('current_user.is_authenticated')
#         return redirect(target_url)

#     form = RegistrationForm()
#     if form.validate_on_submit():
#         print('form validated')

#         # Create a new user instance
#         user = User(username=form.username.data, email=form.email.data)
#         user.set_password(form.password.data)  # Hash and set the password

#         # Add and commit the new user to the database
#         db.session.add(user)
        
#         try:
#             db.session.commit()  # Commit the transaction to save the user
#             flash('Congratulations, you are now a registered user!')
#             # return redirect(url_for('login') + '?target=' + quote_plus(target_url))
#             # Print 
#             print('Redirecting to calibration')
#             return redirect(url_for('calibration'))
#         except Exception as e:
#             db.session.rollback()  # Rollback if there is an error
#             print(f"Error saving user to the database: {e}")
#             flash('An error occurred while trying to register. Please try again.')

#     print('rendering register.html')
#     return render_template('register.html', 
#                            encoded_target=quote_plus(target_url), 
#                            title='Register', 
#                            form=form)

@app.route('/onyenLoginHandler')
def onyen_login_handler():
    env = request.environ

    if 'eppn' in env:
        username = env['eppn'].split('@')[0]
        invite_to_community(username)

    redirect_target = request.args.get('target', '')
    print('----------------Redirect------------------')
    return redirect(redirect_target)

def invite_to_community(username):
    import requests

    # 54 is the group id of group 'pattie_users'
    url = 'https://chipmail.unc.edu/groups/54/members.json'

    payload_json = {
        'usernames': username
    }
    # json serialize
    payload_string = json.dumps(payload_json)

    headers = {
        'Api-Key': os.environ.get('DISCOURSE_API_KEY'),
        'Api-Username': 'system',
        'Content-Type': 'application/json',
    }

    response = requests.request("PUT", url, headers=headers, data = payload_string)

    print(response.text.encode('utf8'))
    
    
def create_directory_in_bucket(bucket_name, directory_name):
    gcloud = current_app.config["GOOGLE_ACCT"]
    credentials = service_account.Credentials.from_service_account_info(gcloud)
    storage_client = storage.Client(credentials=credentials)

    # Get the bucket object
    bucket = storage_client.bucket(bucket_name)

    # The "directory" is an object with a trailing slash
    blob = bucket.blob(f"{directory_name}/")

    # Upload an empty object (directory placeholder)
    blob.upload_from_string("")
