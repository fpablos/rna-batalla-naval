FROM python:3.7
COPY ./src /app
COPY ./requirements.txt /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt