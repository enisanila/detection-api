FROM python:3.9-slim

ENV PYTHONBUFFERED True

ENV APP_HOME /app

WORKDIR $APP_HOME

COPY . ./

RUN pip install -r requirements.txt

# Set env variables for Cloud Run
ENV PORT 8080
ENV HOST 0.0.0.0

# Open port 5000
EXPOSE 8080:8080
# Run flask app
CMD ["python","app.py", "gunicorn"]