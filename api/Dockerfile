FROM python:3.7.8-buster

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
