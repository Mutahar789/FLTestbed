# FLTestbed

## Setup
```
git clone https://github.com/Mutahar789/FLTestbed.git
cd FLTestbed/
./setup.sh
```

## Run Server:
```
./run_server.sh
```

## CPU profiling:
```
./get_cpu_profile.sh <device_id> <device_name>
```

## Install app:
Change IP address in MnistActivity.kt and client variable in LocalFEMNISTDataDataSource.kt <br />
To install the application on a slow device, set the variable isSlowClient to true in LocalFEMNISTDataSource and false otherwise. <br />
Open in android-studio and install. <br />
