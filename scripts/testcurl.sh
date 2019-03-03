#!/bin/bash

curl --header "Content-Type: application/json"   --request POST -d '{"C":3}' http://192.168.188.110:8081/api/train

