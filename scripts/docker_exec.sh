#!/bin/bash

# $1 is the $CONTAINER_ID
# $2 is the shell command to execute
docker exec -ti $1 sh -c "$2"
