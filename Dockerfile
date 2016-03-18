FROM andrewosh/binder-base

MAINTAINER Daniel Wheeler <daniel.wheeler2@gmail.com>

USER root

RUN apt-get update
RUN apt-get install -y git

USER main

## add installation for PyMKS

RUN conda config --add channels wd15
RUN conda install pymks
RUN git clone https://github.com/materialsinnovation/pymks.git
WORKDIR /home/main/pymks

ENV SHELL /bin/bash
