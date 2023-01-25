#! /bin/bash

sudo apt install -y git
sudo snap install android-studio --classic
sudo apt install -y build-essential libssl-dev libffi-dev

sudo apt install python3.7
sudo apt install python3.7-dev
sudo apt install python3.7-venv

python3.7 -m venv fltestbed
source fltestbed/bin/activate

git clone https://github.com/Mutahar789/FLTestbed.git

cd pygrid-federated-feature-federated_process/
pip install poetry
cd apps/node/
poetry install



