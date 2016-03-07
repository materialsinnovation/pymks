FROM andrewosh/binder-base

MAINTAINER Daniel Wheeler <daniel.wheeler2@gmail.com>

USER root

RUN apt-get update
RUN apt-get install -y git

USER main

## add installation for PyMKS

ADD install.yml /home/main/
RUN /home/main/anaconda2/bin/pip install ansible
RUN ansible-playbook install.yml

ENV SHELL /bin/bash
