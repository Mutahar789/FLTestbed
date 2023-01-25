#! /bin/bash

source fl-testbed/bin/activate
cd pygrid-federated-feature-federated_process/apps/node/
./run.sh --id bob --port 5000 --start_local_db
