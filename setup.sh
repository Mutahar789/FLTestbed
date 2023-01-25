#! /bin/bash

sudo snap install android-studio --classic
sudo apt install -y build-essential libssl-dev libffi-dev
sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update
sudo apt install -y python3.7
sudo apt install -y python3.7-dev
sudo apt install -y python3.7-venv

python3.7 -m venv fl-testbed
source fl-testbed/bin/activate

cd pygrid-federated-feature-federated_process/
pip install poetry
cd apps/node/
poetry install

pip install wheel importlib_metadata==4 apscheduler pyfcm flask_apscheduler
pip install matplotlib pandas torch_summary

cd ../../../
cp -r syft $(pip show syft | grep "Location" | grep -oP '(?<=/).*(?=site-packages)' | sed 's/.*/\/&site-packages\//')



