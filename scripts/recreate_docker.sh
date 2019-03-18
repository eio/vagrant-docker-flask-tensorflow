#!/bin/bash

if which vagrant > /dev/null 2>&1
then
	# echo "Vagrant exists."
	if ! vagrant status ellwood-glacier | grep running > /dev/null
	then
		echo "Vagrant is not running. Please run 'vagrant up' first."
		exit 1
	else
		# echo "Vagrant is running."
		vagrant ssh ellwood-glacier -c 'cd /vagrant && docker-compose up --force-recreate -d'
		exit $?
	fi
else
	# echo "Vagrant not in PATH. Attempting to run locally."
	cd /vagrant && docker-compose up --force-recreate -d
fi
