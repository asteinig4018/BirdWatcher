FROM tensorflow/tensorflow
COPY . /conent
WORKDIR /conent
CMD pip install -r requirements.txt