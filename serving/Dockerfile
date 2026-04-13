FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "xgboost>=1.7,<2.0" \
    flask \
    gunicorn

COPY serve.py /opt/program/serve.py

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "3", "--timeout", "120", "serve:app"]
WORKDIR /opt/program
