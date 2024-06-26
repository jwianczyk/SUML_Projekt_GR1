FROM python:3.9-slim

COPY requirements.txt tmp/requirements.txt
RUN python -m pip install --upgrade pip

RUN apt-get update -y
RUN pip install --no-cache-dir -r tmp/requirements.txt && rm -f tmp/requirements.txt


ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]


