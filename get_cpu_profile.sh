#!/bin/bash

while true; do
  adb -s $1 shell cat /proc/stat >> cpu_profile
  echo "###" >> cpu_profile
  sleep 1
done

