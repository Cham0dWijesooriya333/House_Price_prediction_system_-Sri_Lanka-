# Dockerfile_notebook
FROM jupyter/base-notebook

WORKDIR /app

COPY ..
COPY .. requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "server.py"]