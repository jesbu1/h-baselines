#!/bin/bash

DOCKER_HUB=docker.hobot.cc
IMAGE=test/xserver
TAG=0.0.1
docker build -t $DOCKER_HUB/$IMAGE:$TAG -f Dockerfile.gpu .
docker login $DOCKER_HUB
docker push $DOCKER_HUB/$IMAGE:$TAG

