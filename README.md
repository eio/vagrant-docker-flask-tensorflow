# Setup

Install VirtualBox: https://www.virtualbox.org

Install Vagrant: https://www.vagrantup.com/

Install Vagrant plugins:

	vagrant plugin install vagrant-docker-compose
	vagrant plugin install vagrant-disksize

# Running

	vagrant up

^ this will take awhile the first time

# Auto rsync between host and VM

	vagrant rsync-auto

^ this command watcues all files in the host machine's top-level project directory
if a change is made, an automatic rsync will update files in the VM's `/vagrant/` directory

`vagrant rsync` can be used for a one-off file sync

# Verify working

Test POSTs:

	scripts/curl_post_zero.sh

	scripts/curl_post_three.sh

Test GET index in browser:

	http://192.168.188.110:8081

List running Docker containers:

	vagrant ssh
	docker ps

Maybe get shell on a container:

	vagrant ssh
	scripts/shelldock.sh $CONTAINER_ID

# Reference

## Spinning up a VM with Ubuntu and Docker, via Vagrant
https://technology.amis.nl/2018/05/21/rapidly-spinning-up-a-vm-with-ubuntu-and-docker-on-my-windows-machine-using-vagrant-and-virtualbox/

## Simple Docker container + Flask server
https://blog.ouseful.info/2018/08/06/docker-container-and-vagrant-virtualbox-101s/

## Flask server with simple Machine Learning REST API
https://medium.com/@dvelsner/deploying-a-simple-machine-learning-model-in-a-modern-web-application-flask-angular-docker-a657db075280

## React + Redux with Flask backend
https://github.com/YaleDHLab/flask-react-boilerplate

## Docker with scikit-learn

Here we use this:
https://github.com/abn/scipy-docker-alpine/blob/master/Dockerfile

Other options:
https://hub.docker.com/r/buildo/docker-python2.7-scikit-learn/dockerfile
https://hub.docker.com/r/ecoron/python36-sklearn/do
https://hub.docker.com/r/fastgenomics/sklearn/dockerfile

## Docker Compose
https://docs.docker.com/compose/gettingstarted/

## Faster builds

https://medium.com/google-cloud/deploying-scipy-numpy-with-docker-205e9afac3b0

# More Resources

### Basic classification NN with Tensorflow

	https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb

### Saving, loading, and serving Tensorflow models

	https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_restore_models.ipynb?hl=es-MX

### User-drawn MNIST images

	https://codepen.io/MaciejCaputa/pen/KXbJWR


