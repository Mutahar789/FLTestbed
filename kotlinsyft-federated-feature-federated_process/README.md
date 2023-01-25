## Introduction
This repository is a duplicate of [KotlinSyft](https://github.com/OpenMined/KotlinSyft). Please read the detailed instructions there. KotlinSyft downloads the model from [PyGrid](https://github.com/OpenMined/PyGrid/) and executes the training plans hosted on [PyGrid](https://github.com/OpenMined/PyGrid/) using [PySyft](https://github.com/OpenMined/PySyft)

## New Features:
1. Firebase services integration to receive messages from Server
2. Send firebase device tokens to server against each worker
3. Customize authentication pipeline. Save firebase token & intiate cycle requests based on server instruction
4. Connect with the server when user presses submit
5. Upon successful connection, launch MnistActivity and register events for FCM
6. Incorporate EventBus in the App for taking events from Firebase Messaging Service
7. Handle different push notification types from server
8. Start training based on event type
9. Intimate user about the training is finished
10. Change save-firebase-token endpoint to send model name and version
11. Update worker online status upon server request
12. Mechanism to handle cycle start request key. This is required for the case to detect users selected for that cycle should be able to download models etc, but unselected users should be rejected based on cycle_push_request_key
13. Enabling unmetered network conditions
14. Test the model before starting the training
15. Save testing accuracy metrics locally
16. Report the test accuracy metrices on server upon training completion
17. Synthetic dataset loading mechanism
18. Make app configure by setting variable which dataset to run experiment on
19. Run experiment using batch wise data
20. Report number of samples used in training during diff reporting
21. Batch wise data feeding during training
22. Creating evaluation plans for training / testing

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
