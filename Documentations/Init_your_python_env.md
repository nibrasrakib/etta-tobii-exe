1. Install pyenv
https://github.com/pyenv/pyenv
For mac users, I recommend using Homebrew

2. Install pyenv specific python version 3.6.10
```
pyenv install 3.6.10
```

3. Install pyenv-virtualenv, which is a plugin for pyenv that allows you to manange different environments across multiple versions of Python
https://github.com/pyenv/pyenv-virtualenv

4. Create virtualenv with Python version 3.6.10
```
pyenv virtualenv 3.6.10 <your-venv-name>
```

6. Activate your pyenv virtual env 
```
pyenv activate <your-venv-name>
```

5. Install dependencies and you are good to go!
```
pip install -r <requirements-file-name>
```

