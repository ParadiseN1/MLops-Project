FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

WORKDIR /app
RUN mkdir data imgs
RUN mkdir data/models

COPY inference.py ./
COPY Dockerfile ./
COPY requirements.txt ./
COPY data/models/export.pkl ./data/models/
COPY imgs/Beagle_harrier.jpeg ./imgs
COPY start.sh ./

RUN pip install -r requirements.txt
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]


CMD ["python"]
ENTRYPOINT ["start.sh"]