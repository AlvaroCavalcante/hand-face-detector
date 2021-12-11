FROM tensorflow/tensorflow:latest-gpu

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y sudo && \
    apt install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 && \
    pip install -r requirements.txt