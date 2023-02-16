FROM python:3.9
ENV PYTHONUNBUFFERED = 1
WORKDIR classification_model

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
COPY code/requirements.txt /classification_model/
RUN pip install -r requirements.txt

COPY birds_classification_fixed/ /data/input/birds_classification/
ENV IMAGES_DIRECTORY=/data/input/birds_classification/train
RUN mkdir -p /data/modified/birds_classification/train

COPY code/ /classification_model/

CMD python /classification_model/preprocessing.py