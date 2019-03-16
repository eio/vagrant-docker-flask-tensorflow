#!/bin/bash

CONTAINER=$(docker ps -aqf "name=postgres")
docker exec -ti $CONTAINER sh -c "psql -d testdb -U dbuser"
