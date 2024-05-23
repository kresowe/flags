FROM python:3.9

WORKDIR /flags-app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./src/ ./app/

CMD ["python", "./app/recognize_flag.py"]
