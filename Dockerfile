FROM tensorflow/tensorflow
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install libglib2.0-0
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
CMD pip install -r requirements.txt