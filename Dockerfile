FROM python:3.7.12
ENV PYTHONUNBUFFERED = 1
WORKDIR classification_model

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
COPY code/requirements.txt /classification_model/
RUN pip install -r requirements.txt

COPY birds_classification_fixed/train/ /data/raw/
RUN mkdir -p /data/preprocessed/
RUN mkdir -p /model
RUN mkdir -p /inference

COPY code/ /classification_model/

ENTRYPOINT ["python", "/classification_model/main.py"]