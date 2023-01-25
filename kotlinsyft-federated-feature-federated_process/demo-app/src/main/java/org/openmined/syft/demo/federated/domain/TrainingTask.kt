package org.openmined.syft.demo.federated.domain

import android.app.Activity
import android.util.Log
import android.view.View
import androidx.work.ListenableWorker.Result
import io.reactivex.Single
import io.reactivex.processors.PublishProcessor
import org.greenrobot.eventbus.EventBus
import org.greenrobot.eventbus.Subscribe
import org.greenrobot.eventbus.ThreadMode
import org.openmined.syft.Syft
import org.openmined.syft.demo.fcm.PushEvent
import org.openmined.syft.demo.fcm.PushEventType
import org.openmined.syft.demo.federated.logging.MnistLogger
import org.openmined.syft.demo.federated.ui.ContentState
import org.openmined.syft.demo.federated.ui.main.DATASET
import org.openmined.syft.demo.federated.ui.main.SYFT_WORKER_ID
import org.openmined.syft.demo.utils.*
import org.openmined.syft.demo.utils.AppPreferences.Companion.interrupt_training
import org.openmined.syft.domain.SyftConfiguration
import org.openmined.syft.execution.JobStatusSubscriber
import org.openmined.syft.execution.Plan
import org.openmined.syft.execution.SyftJob
import org.openmined.syft.networking.datamodels.ClientConfig
import org.openmined.syft.proto.Placeholder
import org.openmined.syft.proto.SyftModel
import org.openmined.syft.proto.SyftState
import org.pytorch.DType
import org.pytorch.IValue
import org.pytorch.Tensor
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import kotlin.system.exitProcess


class TrainingTask(
    private val configuration: SyftConfiguration,
    private val authToken: String,
    private val datasetType: DATASET = DATASET.MNIST
) {
    private lateinit var mnistDataRepository: MNISTDataRepository
    private lateinit var syntheticDataRepository: SyntheticDataRepository
    private lateinit var femnistDataRepository: FEMNISTDataRepository
    private lateinit var mnistLogger: MnistLogger

    constructor(configuration: SyftConfiguration, authToken: String, mnistDataRepository: MNISTDataRepository, datasetType: DATASET): this(configuration, authToken, datasetType) {
        this.mnistDataRepository = mnistDataRepository
    }

    constructor(configuration: SyftConfiguration, authToken: String, syntheticDataRepository: SyntheticDataRepository, datasetType: DATASET): this(configuration, authToken,datasetType) {
        this.syntheticDataRepository = syntheticDataRepository
    }

    constructor(configuration: SyftConfiguration, authToken: String, femnistDataRepository: FEMNISTDataRepository, datasetType: DATASET): this(configuration, authToken,datasetType) {
//        EventBus.getDefault().register(this)
        this.femnistDataRepository = femnistDataRepository
    }

    private lateinit var syftWorker: Syft

    fun runTask(logger: MnistLogger): Single<Result> {
        if (!::syftWorker.isInitialized)
            syftWorker = Syft.getInstance(configuration, authToken)

        if (!::mnistLogger.isInitialized)
            this.mnistLogger = logger

        var completed = false
        var cycleStartPushKey = getCycleStartPushKey()
        val mnistJob = syftWorker!!.newJob(MODEL_NAME, MODEL_VERSION)
        mnistJob.setCycleStartRequestKey(cycleStartPushKey)
        val statusPublisher = PublishProcessor.create<Result>()

//        logger.postLog("MNIST job started \n\nChecking for download and upload speeds")
//        logger.postLog("Dataset: ${datasetType}")
//        logger.postState(ContentState.Loading)
        val jobStatusSubscriber = object : JobStatusSubscriber() {
            override fun onReady(
                model: SyftModel,
                cycleId: String,
                plans: ConcurrentHashMap<String, Plan>,
                clientConfig: ClientConfig
            ) {
                setTrainingMode(true)

                logger.postLog("Model ${model.modelName} received.\n\nStarting training process")
                trainingProcess(this, mnistJob, model, cycleId, plans, clientConfig, mnistLogger)
            }

            override fun onComplete() {
                Log.e("TrainingTask", "--------------------------- onComplete")

                setTrainingMode(false)

                statusPublisher.offer(Result.success())
                AppPreferences.putBoolean(interrupt_training, false)
                if(AppPreferences.getBoolean(interrupt_training)) {
                    AppPreferences.putBoolean(interrupt_training, false)
//                    EventBus.getDefault().post(PushEvent(PushEventType.CYCLE_REQUEST))

                } else {
                    setCycleStartPushKey("")
                }
            }

            override fun onComplete(reportTime: Long, cycleId: String) {
//                val reportTime =  "$cycleId:$reportTime"
//                Log.e("TrainingTask", "--------------------------- onComplete2 for different times $reportTime")
//                AppPreferences.putString("report_time", reportTime)
            }

            override fun onRejected(timeout: String) {
                setTrainingMode(false)
                setCycleStartPushKey("")
//                logger.postLog("We've been rejected for the time $timeout")
//                logger.postState(ContentState.REJECTION)
                mnistJob.markJobCompleted()
                statusPublisher.offer(Result.retry())
            }

            override fun onError(throwable: Throwable) {
                setTrainingMode(false)
                setCycleStartPushKey("")
//                logger.postLog("There was an error $throwable")
//                 logger.postLog("Stack: ${throwable.printStackTrace()}")
                statusPublisher.offer(Result.failure())
            }
        }
        mnistJob.start(jobStatusSubscriber)

        return statusPublisher.onBackpressureBuffer().firstOrError()
    }

    fun disposeTraining() {
        syftWorker?.dispose()
    }

    private fun trainingProcess(
        jobStatusSubscriber: JobStatusSubscriber,
        mnistJob: SyftJob,
        model: SyftModel,
        cycleId: String,
        plans: ConcurrentHashMap<String, Plan>,
        clientConfig: ClientConfig,
        logger: MnistLogger
    ) {
        val testAccuracyList = test_model(model,plans)
        femnistDataRepository.clearData()
        train_model(jobStatusSubscriber, mnistJob, model, cycleId, plans, clientConfig, testAccuracyList, logger)

    }

    /**
     * Function that executes the Training plan obtained from server
     *
     */
    private fun train_model(jobStatusSubscriber: JobStatusSubscriber,
                            mnistJob: SyftJob,
                            model: SyftModel,
                            cycleId: String,
                            plans: ConcurrentHashMap<String, Plan>,
                            clientConfig: ClientConfig,
                            testAccuracy: ArrayList<Pair<Float, Int>>,
                            logger: MnistLogger) {
        var loss = -0.0f
        var accuracy = -0.0f
        var steps = 1
        val seed = clientConfig.planArgs["seed"]!!.toLong()
        val optimizerStr = clientConfig.planArgs["optimizer"]!!
        val optimizer = optimizerStr.substring(1, optimizerStr.length-1)
        val cycle_timeout = clientConfig.planArgs["cycle_length"]!!.toInt()
        val serverRoundTimeInMillis = cycle_timeout * 1000

        val currentTime = System.currentTimeMillis()
        val trainingEndTime = currentTime + (serverRoundTimeInMillis - 2 * 1000)

        val bootstrap_rounds = clientConfig.planArgs["bootstrap_rounds"]!!.toInt()
        var activationState: SyftState? = null
        AppPreferences.putBoolean(interrupt_training, false)
        femnistDataRepository.initDataReaders()
        Timber.d(">>>>>>>>>>>>>>>>>>>  Cycle ${cycleId}")
        val trainTestSample = getTrainTestSampleCount(false)

        Timber.d(">>>>>>>>>>>>>>>>>>>>>>> trainTestSamples train: ${trainTestSample.first}, test: ${trainTestSample.second}")
        Timber.e("optimizer $optimizer")

        var totalTrainTimeStart = System.currentTimeMillis()
        val totalEpochs = clientConfig.properties.maxUpdates
        var trainedEpochs = 0
        plans["training_plan"]?.let { plan ->
            run repeatBlock@{
                repeat(totalEpochs) { step ->
                    val epochTimeStart = System.currentTimeMillis()
                    trainedEpochs = step + 1
                    logger.postEpoch("Cycle: $cycleId, Epochs: ${step + 1}")
                    val batchSize = (clientConfig.planArgs["batch_size"]
                        ?: error("batch_size doesn't exist")).toInt()
                    val batchIValue = IValue.from(
                        Tensor.fromBlob(longArrayOf(batchSize.toLong()), longArrayOf(1))
                    )
                    val lr = IValue.from(
                        Tensor.fromBlob(
                            floatArrayOf(
                                (clientConfig.planArgs["lr"] ?: error("lr doesn't exist")).toFloat()
                            ),
                            longArrayOf(1)
                        )
                    )

                    if (datasetType == DATASET.SYNTHETIC) {
                        syntheticDataRepository.setIsAllBatchConsumed(false)
                        syntheticDataRepository.setIsRandomizationRequired(true)
                        syntheticDataRepository.setSeed(seed)
                        var count = 0
                    } else if (datasetType == DATASET.FEMNIST) {
                        femnistDataRepository.setIsAllBatchConsumed(false)
                        femnistDataRepository.setIsRandomizationRequired(true)
                        femnistDataRepository.setSeed(seed)
                        var count = 0
                    }


                    while (!((datasetType == DATASET.SYNTHETIC && syntheticDataRepository.getIsAllBatchConsumed()) || (datasetType == DATASET.FEMNIST && femnistDataRepository.getIsAllBatchConsumed()))) {
                        if (isTrainingInterrupted(trainingEndTime)  && (optimizer == "fedavg" || optimizer == "fedprox")) {
                            Timber.e("FedProx condition true")
                            trainedEpochs -= 1
                            return@repeatBlock
                        }

                        var batchData: Pair<IValue, IValue> = when (datasetType) {
                            DATASET.SYNTHETIC -> getBatchData(
                                batchSize,
                                plans["convert_to_one_hot_plan"]
                            )

                            DATASET.FEMNIST -> getBatchData(
                                batchSize,
                                plans["convert_to_one_hot_plan"]
                            )

                            else -> getBatchData(batchSize, null)
                        }

                        val modelParams = model.paramArray ?: return
                        val paramIValue = IValue.listFrom(*modelParams)

                        val output = plan.execute(
                            batchData.first,
                            batchData.second,
                            batchIValue,
                            lr, paramIValue
                        )?.toTuple()
                        output?.let { outputResult ->
                            val paramSize = model.stateTensorSize!!
                            var beginIndex = outputResult.size - paramSize

                            if (cycleId.toInt() <= bootstrap_rounds) {
                                val activations = outputResult[3]
                                activationState = SyftState.createSyftState(
                                    activations,
                                    configuration,
                                    plans["sum_activations"]!!
                                )
                            }

                            beginIndex = 4
                            val updatedParams =
                                outputResult.slice(beginIndex until outputResult.size)
                            model.updateModel(updatedParams)
                        }

//                        return@repeatBlock
                    }

                    if (cycleId.toInt() <= bootstrap_rounds) {
                        SyftState.performAvgOverSamples(
                            plans["average_activations"]!!,
                            plans["sum_activations"]!!,
                            trainTestSample.first
                        )
                    }

                    Timber.e("Epoch ${steps + 1} train time is ${System.currentTimeMillis() - epochTimeStart}")
                    steps++

                }
            }

            femnistDataRepository.clearData()

            if(optimizer != "fedavg") {
                if (cycleId.toInt() <= bootstrap_rounds) {
                    activationState = SyftState.performAvgOverEpochs(
                        plans["average_activations"]!!,
                        totalEpochs
                    )
                }
            }
            logger.postLog("Training done!\n reporting diff")

//            val diff = mnistJob.createDiff()

            mnistJob.updateModelUpdate("${configuration.filesDir}/models/updatedmodel.pb")

//            val modelUpdateFile = File("${configuration.filesDir}/models/updatedmodel.pb")
//            val uploadFileSizeInBytes = modelUpdateFile.length()



            val totalNumSamples = trainTestSample.first + trainTestSample.second
            val modelFile = File("${configuration.filesDir}/models/${model.pyGridModelId}.pb")
            val modelFileSizeInBytes = modelFile.length()

//            Timber.e(">>>>>>>>>>>>>>>>>> uploadFileSizeInBytes ${(uploadFileSizeInBytes / 1024)/1024} Download Size: ${(modelFileSizeInBytes / 1024)/1024}")

            val trainEndTime = System.currentTimeMillis()
            val totalTimeMillis =  trainEndTime - totalTrainTimeStart

            val totalTrainTime = totalTimeMillis

            val avgEpochTrainTime = totalTrainTime / totalEpochs

            if(isTrainingInterrupted(trainingEndTime) && optimizer == "fedavg") {
                Timber.e("Reporting stats only")
                AppPreferences.putBoolean(interrupt_training, false)
                mnistJob.reportTrainingStats(
                    testAccuracy,
                    modelFileSizeInBytes,
                    avgEpochTrainTime,
                    totalTrainTime,
                    cycleId
                )
            }
            else {
                Timber.e("Reporting parameters")

                var modelDownloadTime = ""
                var modelUploadTime = ""
                var modelDownloadSize = ""
                var modelUploadSize = ""

                modelDownloadTime = "${(AppPreferences.getString("download_time_end").toDouble() - AppPreferences.getString("download_time_start").toDouble()) / 1000}"
                modelDownloadSize = "${AppPreferences.getString("model_download_size").toInt()}"
                if(AppPreferences.getString("report_time_end").length > 0) {
                    modelUploadTime = "${(AppPreferences.getString("report_time_end").toDouble() - AppPreferences.getString("report_time_start").toDouble()) / 1000}"
                    modelUploadSize = "${AppPreferences.getString("model_upload_size").toInt()}"
                }

                activationState?.let { activations ->
                    mnistJob.reportWithActivations(jobStatusSubscriber,
                        mnistJob.getModelState(),
                        testAccuracy,
                        totalNumSamples,
                        modelFileSizeInBytes,
                        avgEpochTrainTime,
                        totalTrainTime,
                        cycleId,
                        trainedEpochs,
                        "$modelDownloadTime",
                        "$modelUploadTime",
                        "$modelDownloadSize",
                        "$modelUploadSize",
                        activations
                    )
                } ?: mnistJob.report(
                    jobStatusSubscriber,
                    mnistJob.getModelState(),
                    testAccuracy,
                    totalNumSamples,
                    modelFileSizeInBytes,
                    avgEpochTrainTime,
                    totalTrainTime,
                    cycleId,
                    trainedEpochs,
                    "$modelDownloadTime",
                    "$modelUploadTime",
                    "$modelDownloadSize",
                    "$modelUploadSize"
                )
            }

            if(!isTrainingInterrupted(trainingEndTime)) {
                setCycleStartPushKey("")
            }
            AppPreferences.putBoolean(interrupt_training, false)
            activationState = null
            SyftState.resetOldSyftState()
            setTrainingMode(false)
        }
    }

    fun isTrainingInterrupted(trainingEndTime: Long) = System.currentTimeMillis() >= trainingEndTime && femnistDataRepository.is_slow_client()

    /**
     * Load the batch data for training
     */
    private fun getBatchData(batchSize: Int,plan: Plan?, isTest: Boolean = false): Pair<IValue, IValue> {

        when(datasetType) {
            DATASET.SYNTHETIC -> {
                if(isTest) {
                    return syntheticDataRepository.loadTestDataBatch(plan!!, isTest)
                } else {
                    return syntheticDataRepository.loadDataBatch(batchSize, plan!!)
                }
            }

            DATASET.FEMNIST -> {
                if(isTest) {
                    return femnistDataRepository.loadTestDataBatch(plan!!, isTest)
                } else {
                    return femnistDataRepository.loadDataBatch(batchSize, plan!!)
                }
            }

            else -> {
                return  mnistDataRepository.loadDataBatch(batchSize)
            }
        }
    }

    /**
     * Load the batch data for evaluation
     */
    private fun getTestData(batchSize: Int, plan: Plan?, forTrainData: Boolean = false): Pair<IValue, IValue> {

        when(datasetType) {
            DATASET.SYNTHETIC -> {
                return syntheticDataRepository.loadTestBatch(plan!!, forTrainData)
            }

            DATASET.FEMNIST -> {
                return femnistDataRepository.loadTestBatch(plan!!, forTrainData)
            }

            else -> {
                return  mnistDataRepository.loadDataBatch(batchSize)
            }
        }
    }

    /**
     * Get the training and testing sample count
     */
    private fun getTrainTestSampleCount(isTest: Boolean = false): Pair<Int, Int> {

        when(datasetType) {
            DATASET.SYNTHETIC -> {
                return syntheticDataRepository.getTotalNumberOfSamples()
            }

            DATASET.FEMNIST -> {
                return femnistDataRepository.getTotalNumberOfSamples()
            }

            else -> {
                return  mnistDataRepository.getTotalNumberOfSamples()
            }
        }
    }



    /**
     * Function that executes the testing plan.
     *
     */
    private fun test_model(model: SyftModel, plans: ConcurrentHashMap<String, Plan>): ArrayList<Pair<Float, Int>> {

        plans["evaluate_model_plan"]?.let {plan ->
            val oneHotVectorPlan = plans["convert_to_one_hot_plan"]!!

            val accuracy_list: ArrayList<Pair<Float, Int>> = arrayListOf()


            val modelParams = model.paramArray
            val paramIValue = IValue.listFrom(*modelParams!!)

            var batchData: Pair<IValue, IValue>? = null

            val test_ids = femnistDataRepository.get_test_data_ids()
            val train_ids = femnistDataRepository.get_train_data_ids()

            test_ids.forEachIndexed { index, id ->
                val testDataId = id
                val trainDataId = train_ids[index]

                batchData = femnistDataRepository.loadTestDataBatch(plan,oneHotVectorPlan, id, false)
                val trainTestSample = femnistDataRepository.getTotalNumberOfSamples(trainDataId, testDataId)
                var batchIValue = IValue.from(Tensor.fromBlob(longArrayOf(trainTestSample.second.toLong()), longArrayOf(1)))
                val output = plan.execute(
                    batchData!!.first,
                    batchData!!.second,
                    batchIValue,
                    paramIValue
                )?.toTuple()

                val totalSum = trainTestSample.first + trainTestSample.second
                accuracy_list.add(Pair(output!![0].toTensor().dataAsFloatArray.last(), totalSum))
            }

            return accuracy_list
        } ?: return arrayListOf(Pair(0f, 0))

    }
}