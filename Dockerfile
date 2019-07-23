## ==================== Final layer ======================

FROM python:3.6-slim-stretch

RUN apt-get -qq update &&\
    apt-get install -y \
    curl git \
    &&\
    apt-get install --no-install-recommends -y \
    libglib2.0-0 \
    libavcodec-dev \
    libavformat-dev \
    && \
    apt-get clean && rm -rf /tmp/* /var/tmp/*

#COPY --from=cv_deps /root/diff /

COPY . /root/faced

RUN pip3 install --no-cache-dir \
    opencv-python-headless \
    tensorflow \
    /root/faced
