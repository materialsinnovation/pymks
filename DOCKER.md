# Run PyMKS in a Docker Instance

## Install Docker

Install Docker and run the Daemon. See
https://docs.docker.com/engine/install/ for installation
details. Launch the Daemon if necessary.

    $ sudo service docker start

## Pull the Docker instance

Pull the Docker Instance from Dockerhub

    $ docker pull docker.io/wd15/pymks

## Run the Tests

Run the tests.

    $ docker run -i -t wd15/pymks:latest
    # python -c "import pymks; pymks.test()"

## Use PyMKS in a Jupyter notebook

    $ docker run -i -t -p 8888:8888 wd15/pymks:latest
    # jupyter notebook --ip 0.0.0.0 --no-browser

The PyMKS example notebooks are availabe inside the image after
opening the notebook using http://127.0.0.1:8888.

## Build the Docker instance

Clone this repository and build the instance.

    $ docker load < $(nix-build docker.nix)

or to build and then launch use

    $ docker load < $(nix-build docker.nix) && docker run -p 8888:8888 -it wd15/pymks:latest

## Push the Docker instance

Create the repository in Dockerhub and then push it.

    $ docker login
    $ docker push docker.io/wd15/pymks
