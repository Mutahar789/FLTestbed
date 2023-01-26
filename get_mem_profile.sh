#! /bin/bash

cd pygrid-federated-feature-federated_process/examples/model-centric/data/

# Nexus 6P
adb -s 84B5T15B03015109 shell "su -c cat /data/user/0/org.openmined.syft.demo/files/mydir/memoryinfo.csv" > "Nexus 6P_memLog.csv"

# Nexus 5x
adb -s 00f2aa907991b4e1 shell "su -c cat /data/user/0/org.openmined.syft.demo/files/mydir/memoryinfo.csv" > "Nexus 5x_memLog.csv"

# Nexus 5
adb -s 0ba3223902e1a4b4 shell "cat /data/user/0/org.openmined.syft.demo/files/mydir/memoryinfo.csv" > "Nexus 5_memLog.csv"

# Nokia 1
adb -s FRTBA80314746200 shell "su -c cat /data/user/0/org.openmined.syft.demo/files/mydir/memoryinfo.csv" > "Nokia 1_memLog.csv"