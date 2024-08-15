FROM mambaorg/micromamba:1.5.8
# FROM python:3.11

# you need git installed for installing repos with pip
USER root
# ENV http_proxy=deb.debian.org/debian
# ENV https_proxy=deb.debian.org/debian
RUN apt-get update --fix-missing && apt-get install -y git

COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.yml /tmp/env.yaml

RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes


# Activate the environment and install pip, gunicorn, and requests
RUN micromamba run -n base pip install gunicorn requests && \
    micromamba run -n base gunicorn --version && \
    micromamba run -n base python -c "import requests; print(requests.__version__)"


ARG MAMBA_DOCKERFILE_ACTIVATE=1 

COPY ./app /code/app

WORKDIR /code

ENV PYTHONPATH=/code

EXPOSE 8000
# need to use double quote here!
#CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["gunicorn", "-b", "0.0.0.0:8000", "app.server:app"]
CMD ["gunicorn", "-b", "0.0.0.0:80", "app.server:app", "-k", "uvicorn.workers.UvicornWorker"]


