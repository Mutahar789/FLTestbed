## Introduction

This repository is duplicate of [PyGrid](https://github.com/OpenMined/PyGrid/). Please read the original repository for detailed instructions on how to setup PyGrid and relevant component and for hosting the model centric federated learning (FL). Brief introduction of model centric FL is as follows:


#### Model-centric FL

Model-centric FL is when the model is hosted in PyGrid. This is really useful when you have data located at an "edge device" like a person's mobile phone or web browser. Since the data is private, we should respect that and leave it on the device. The following workflow will take place:

1. The device will request to train a model
2. The model and a training plan may be sent to that device
3. The training will take place with private data on the device itself
4. Once training is completed, a "diff" is generated between the new and the original state of the model
5. The diff is reported back to PyGrid and it's averaged into the model

This takes place potentially with hundreds, or thousands of devices simultaneously. **For model-centric federated learning, you only need to run a Node. Networks and Workers are irrelevant for this specific use-case.**

### Workflow after adding New Features to PyGrid look like below:

![alt text](https://github.com/mustansarsaeed/pygrid-federated/blob/feature/federated_process/assets/Workflow-Github.png)


## New Features:
1. Save Firebase tokens against each worker 
2. Send firebase messages to workers if enough pool of workers available
3. Add warehouse the capability to join multiple columns
4. Check if enough pool of workers available for sending cycle request command
5. Create function and endpoint to send the push notification to different devices
6. Find all those workers who are assigned to any cycle and that cycle has not yet completed
7. Integrate pyfcm to send the notifications in synchronous way
8. Create common trigger event that can be used anywhere in the codebase by importing the module
9. Send trigger events based whenever user connects with the server
10. Instruct workers to request for the new cycle when cycle gets completed
11. Worker participation mode configuration added.
12. Ask workers to update their availability after cycle completion & new user registration in case no cycle is started yet
13 Mechansim to detect that training has been started.
14. If user authenticates with the server then if training is already started then don't send push for next training
15. If cycle gets expired then reset the cycle, delete workers, send workers notification to update availability and next cycle
16. Random selection of users using random sampling
17. Create Pan notebook changes to read the testing stats from database and generate plot of accuracy vs round number
18. Endpoint for accepting the testing metrics from the clients and save into the database
19. FCM Push Manager module: When cycle not found, send push to workers that training is completed
20. Plans are not being loaded tfjs issue fixed
21. LEAF Averaging formula
22. Save num of samples against each diff when client reports
23. Implement leaf aggregation algorithm

## Upcoming Features:
1. Autograd options for Sigmoid and CNN
2. Model pruning
