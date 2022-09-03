#!/bin/bash

curl -H 'Content-Type: application/json' -X POST --data '{"message": "não confio nesse governo"}' localhost:5000/predict
