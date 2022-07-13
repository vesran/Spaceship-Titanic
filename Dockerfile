FROM python:3.10-slim

WORKDIR /app

RUN apt-get install && apt-get update
RUN python -m pip install --upgrade pip
RUN apt-get -y install curl
RUN apt install build-essential -y --no-install-recommends
RUN apt-get install swig python3-dev -y

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app/src/"

CMD ["/bin/bash"]