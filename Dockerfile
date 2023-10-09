FROM python:3.9-slim@sha256:980b778550c0d938574f1b556362b27601ea5c620130a572feb63ac1df03eda5 

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

ENV PORT 8008

RUN pip install --no-cache-dir -r requirements.txt \
    bash setup.sh \ 
    pip install -U jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    pip install -U jaxlib==0.4.12+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1