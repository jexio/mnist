FROM fulmo:1.0.0
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workdir

RUN python3.8 -m pip install /workdir/fulmo/
COPY requirements.txt /workdir
RUN python3.8 -m pip install -r /workdir/requirements.txt --no-cache-dir
