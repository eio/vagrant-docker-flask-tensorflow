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
		vagrant ssh ellwood-glacier -c 'docker exec -ti $(docker ps -aqf "name=postgres") sh -c "psql -d testdb -U dbuser"'
		exit $?
	fi
else
	# echo "Vagrant not in PATH. Attempting to run locally."
	CONTAINER=$(docker ps -aqf "name=postgres")
	docker exec -ti $CONTAINER sh -c "psql -d testdb -U dbuser"
fi
