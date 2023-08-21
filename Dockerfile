FROM python:3.9-slim@sha256:980b778550c0d938574f1b556362b27601ea5c620130a572feb63ac1df03eda5 

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

ENV PORT 8008

RUN pip install --no-cache-dir -r requirements.txt

CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1