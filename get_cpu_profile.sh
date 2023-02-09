#!/bin/bash

while true; do
  adb -s $1 shell cat /proc/stat >> ./pygrid-federated-feature-federated_process/examples/model-centric/data/$2_cpu_profile.txt
  echo "###" >> ./pygrid-federated-feature-federated_process/examples/model-centric/data/$2_cpu_profile.txt
  sleep 1
done

