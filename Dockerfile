# Set base image (host OS)
FROM python:3.8-slim
# By default, listen on port 5000
EXPOSE 5000/tcp

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install Cython
RUN \ 
    apt-get update && \
    apt-get -y install libpq-dev gcc build-essential
RUN pip install -r requirements.txt 
# Copy the content of the local src directory to the working directory
COPY . /app
# ENV FLASK_APP=main
# Specify the command to run on container start
# CMD flask run --host=0.0.0.0
# CMD [ "python", "./app.py" ]
ENTRYPOINT [ "python" ]

CMD ["plos_exp.py" ]