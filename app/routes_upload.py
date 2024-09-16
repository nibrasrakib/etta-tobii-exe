from app import app
from app.utils import allowed_file, check_size
import os
from flask import render_template, redirect, url_for, request, flash
from flask_login import current_user
import numpy as np
import chardet
from werkzeug.utils import secure_filename
from urllib.parse import quote_plus

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    '''
    if request.method == "POST" and 'inout_text' in request.files:
        filename = text.save(request.files['input_text'])
        result = "File has been uploaded successfully!"
        return render_template('upload.html', result = result)
    return render_template('upload.html', result = "")
    '''
    env = request.environ

    # if onyen login
    if 'eppn' in env:
        username = env['eppn'].split('@')[0]
        path = os.path.join(app.config['UPLOAD_FOLDER'], username)
    # if general login
    elif current_user.is_authenticated:
        username = current_user.email
        path = os.path.join(app.config['UPLOAD_FOLDER'], username)
    else:
        path = app.config['UPLOAD_FOLDER']
        return redirect(url_for('login') + '?target=' + quote_plus(request.base_url))

    # app.config['UPLOAD_FOLDER'] = path

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
#       delimiter = request.form['delimiter']
        delimiter = ''
        if delimiter == '':
            delimiter = 'no delimiter provided'
        print(delimiter)

        count = 0
        for r, d, f in os.walk(path):
            for file_ in f:
                if '.npy' in file_:
                    count += 1
        if count > 0:
            print('found .nyp')
            # Load
            del_menu = np.load(os.path.join(path, 'del_menu.npy'),
                               allow_pickle=True).item()
            print(del_menu)
            # change
            del_menu[file.filename] = delimiter
            print(del_menu)
            # Save
            np.save(os.path.join(path, 'del_menu.npy'), del_menu)
        else:
            del_menu = dict()
            del_menu[file.filename] = delimiter
            np.save(os.path.join(path, 'del_menu.npy'), del_menu)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file:
            if not allowed_file(file.filename):
                return render_template('upload.html', result="", warning="Looks like your file has the wrong format? Please only upload .txt and .csv files. ")
            if not check_size(file):
                print("Invalid file size")
                warning = "File is too large. Please upload file smaller than " + \
                    str((ALLOWED_FILE_SIZE/1024)/1024) + " MB. "
                return render_template('upload.html', result="", warning=warning)

            filename = secure_filename(file.filename)
            print("Valid file size")
            filepath = os.path.join(path, filename)
            if os.path.exists(filepath):
                flash("Looks like you've already upload a file with the same name. Please upload a file with a different name or delete the original one first. ")
                return redirect(request.url)
            file.save(filepath)
            file = open(filepath, 'rb')
            text = file.read()
            print(filename)
#           print("text is :\n")
#           print(text)
            encoding = chardet.detect(text)
            encoding = encoding['encoding']
            print(encoding)
            text = text.decode(encoding)
            file = open(os.path.join(path, filename), 'w', encoding="utf-8")
            file.write(text)
            '''
            print(type(file))
            file = file.read()
            print(type(file))
            file_list = file.split(divider)
            with open(filename, 'w') as f:
                for item in file_list:
                    f.write("\n********\n" + item)
            '''
            return render_template('upload.html', result="File uploaded successfully.", warning="")
    return render_template('upload.html', result="", warning="")

@app.route('/delete')
def delete_file():
    env = request.environ

    file_name = request.args.get('file', '')
    # rule out other dataset options
    if '.txt' not in file_name and '.csv' not in file_name:
        print('-----------WRONG FILE TYPE: ' + file_name + '-----------')
    else:
        # if onyen login
        if 'eppn' in env:
            # route to the user-specific folder
            username = env['eppn'].split('@')[0]
            path_to_file = os.path.join(app.config['UPLOAD_FOLDER'], username, file_name)
        # if general login
        elif current_user.is_authenticated:
            username = current_user.email
            path_to_file = os.path.join(app.config['UPLOAD_FOLDER'], username, file_name)
        else:
            print('-----------SHOULD NEVER REACH HERE-----------')
            path_to_file = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            return redirect(url_for('login') + '?target=' + quote_plus(url_for('home')))

        if os.path.exists(path_to_file):
            print('remove file at: ' + path_to_file)
            os.remove(path_to_file)
        else:
            print('The file does not exist')

    return redirect(url_for('home'))
