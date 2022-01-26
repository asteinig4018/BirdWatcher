FROM tensorflow/tensorflow
COPY . /app
WORKDIR /app
CMD pip install -r requirements.txt