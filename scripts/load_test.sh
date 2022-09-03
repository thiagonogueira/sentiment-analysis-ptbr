#!/bin/bash

K8S_IP="$1"
K8S_PORT="$2"

if [[ -z "${K8S_IP}" || -z "${K8S_PORT}" ]] ; then
    echo -e "\nDois parametros necessarios: IP publico para inferencia e porta\n"
    echo -e "Exemplo de chamada ./load_test.sh 34.111.222.1 5000\n"
    exit 1
fi

echo -e "\nIniciando teste de estresse\n"
echo -e "URL: http://${K8S_IP}:${K8S_PORT}/predict\n"
sleep 5
while true
do
    curl -H 'Content-Type: application/json' -X POST --data '{"message": "não confio nesse governo"}' http://${K8S_IP}:${K8S_PORT}/predict
done
