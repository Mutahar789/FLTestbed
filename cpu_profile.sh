#! /bin/bash

cd pygrid-federated-feature-federated_process/examples/model-centric/data/

# Nexus 6P
adb -s 84B7N15A07007998 shell top | grep openmined > "Nexus 6P_cpu_info"

# Nexus 5x
adb -s 00f2aa907991b4e1 shell top | grep openmined > "Nexus 5x_cpu_info"

# Nexus 5
adb -s 0ba3223902e1a4b4 shell top | grep openmined > "Nexus 5_cpu_info"

# Nokia 1
adb -s FRTBA80314746200 shell top | grep openmined > "Nokia 1_cpu_info"
