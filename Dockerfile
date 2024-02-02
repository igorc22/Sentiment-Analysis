FROM python:3.10 
COPY . /app
WORKDIR /app
RUN pip install -r Requirements.txt
EXPOSE 8080
CMD gunicorn --workers=4 --bind 0.0.0.0:8080 app:app