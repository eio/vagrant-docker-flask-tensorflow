FROM debian:latest

COPY . /code
WORKDIR /code

RUN apt-get update \
    && apt-get -y install python3-pip \
    && pip3 --version \
    && pip3 install --no-cache-dir -r requirements.txt \
    && apt-get -y install vim

CMD ["python3", "app/app.py"]
