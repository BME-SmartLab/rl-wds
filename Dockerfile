FROM python:3.7-slim

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV NUM_PROCS=1
ENV BOKEH_RESOURCES=cdn

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get install libgtk2.0-dev -y \
    && apt-get clean

WORKDIR /demo

COPY ./requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./ .

EXPOSE 8080

CMD bokeh serve --port 8080 \
    --allow-websocket-origin="*" \
    --num-procs=${NUM_PROCS} \
    ./demo.py
