# About

This project uses Vagrant to spin up a virtual machine running Debian.

It provisions that Debian VM with Docker.

The Debian VM then creates a Debian Docker image which serves a simple website and REST API via the Flask framework for Python.

The website allows users to draw a digit on an HTML canvas.

The website then scales the hand-drawn digit to the size of images from the MNIST dataset of hand-drawn digits.

The scaled image is then base64-encoded and sent to the server to be checked against a neural network model which was previously trained against the MNIST dataset of hand-drawn digits.

The server responds with JSON data containing a prediction and confidence level:

```
{ 'prediction': '4', 'confidence': '80%' }
```

The website informs the user of the results.

# Setup

Install VirtualBox: https://www.virtualbox.org

Install Vagrant: https://www.vagrantup.com/

Install Vagrant plugins:

	vagrant plugin install vagrant-docker-compose
	vagrant plugin install vagrant-disksize

# Running

	vagrant up

^ this will take awhile the first time

# Verify things are working

Test POSTs:

	./scripts/test_post_zero.sh

	./scripts/test_post_three.sh

Test GET index in browser:

	http://192.168.188.110:8081

List running Docker containers from the VM:

	vagrant ssh
	docker ps

Maybe get shell on a container through the VM:

	vagrant ssh
	scripts/bash_docker.sh $CONTAINER_ID

# Automatically sync files between host and VM

	vagrant rsync-auto

^ this command watches all files in the host machine's top-level project directory (as specified by the `config.vm.synced_folder` line in the `Vagrantfile`)

if a change is made, an automatic rsync will update files in the VM's `/vagrant/` directory, thus ensuring that both directories (the host machine's and the VM's) contain the same content.

`vagrant rsync` can be used for a one-off file sync

# Resources

### Spinning up a VM with Ubuntu and Docker, via Vagrant
https://technology.amis.nl/2018/05/21/rapidly-spinning-up-a-vm-with-ubuntu-and-docker-on-my-windows-machine-using-vagrant-and-virtualbox/

### Simple Docker container + Flask server
https://blog.ouseful.info/2018/08/06/docker-container-and-vagrant-virtualbox-101s/

### Flask server with simple Machine Learning REST API
https://medium.com/@dvelsner/deploying-a-simple-machine-learning-model-in-a-modern-web-application-flask-angular-docker-a657db075280

### React + Redux with Flask backend
https://github.com/YaleDHLab/flask-react-boilerplate

### Docker Compose
https://docs.docker.com/compose/gettingstarted/

### Faster builds
https://medium.com/google-cloud/deploying-scipy-numpy-with-docker-205e9afac3b0

### Basic classification NN with Tensorflow
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb

### Saving, loading, and serving Tensorflow models
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_restore_models.ipynb?hl=es-MX

### User-drawn MNIST images
https://codepen.io/MaciejCaputa/pen/KXbJWR

### Deep Convolutional Generative Adversarial Networks with TensorFlow
https://github.com/dmonn/dcgan-oreilly/blob/master/DCGANs%20with%20Tensorflow.ipynb
