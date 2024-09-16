# for user login
from flask_login import current_user, login_user, logout_user
from flask import render_template, redirect, url_for, request, flash, json
from urllib.parse import quote_plus
from app import app, db
from app.forms import LoginForm, RegistrationForm
from app.models import User
import os

@app.route('/login', methods=['GET', 'POST'])
def login():
    env = request.environ
    target_url = request.args.get('target')
    print(target_url)

    if 'eppn' in env:
        return redirect(target_url)
    if current_user.is_authenticated:
        return redirect(target_url)
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('No such user or the password is incorrect. ')
            return redirect(url_for('login') + '?target=' + quote_plus(target_url))
        login_user(user, remember=form.remember_me.data)
        return redirect(target_url)
    return render_template('login.html', encoded_target=quote_plus(target_url), titel='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    target_url = request.args.get('target')
    return redirect(target_url)


@app.route('/register', methods=['GET', 'POST'])
def register():
    env = request.environ
    target_url = request.args.get('target')
    print(target_url)

    if 'eppn' in env:
        return redirect(target_url)
    if current_user.is_authenticated:
        return redirect(target_url)

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login') + '?target=' + quote_plus(target_url))
    return render_template('register.html', encoded_target=quote_plus(target_url), title='Register', form=form)

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
