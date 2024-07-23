FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 \
    && rm -rf /var/lib/apt/lists/*

##### Set workdir in docker
RUN mkdir /workdir
WORKDIR /workdir

##### Dependencies
COPY requirements.txt /workdir/requirements.txt
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && python3.10 -m pip install -r requirements.txt \
    && python3.10 -m pip install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118

##### Copy code
COPY . /workdir/
RUN echo $(ls)

RUN chown -R 1000:root /workdir && chmod -R 775 /workdir

ENTRYPOINT python3 run_warmup.py && python3 run_ppo.py
