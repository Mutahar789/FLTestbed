package org.openmined.syft.demo.federated.ui.main

import android.accounts.NetworkErrorException
import android.app.Activity
import android.system.ErrnoException
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModel
import androidx.work.WorkInfo
import io.reactivex.disposables.CompositeDisposable
import org.openmined.syft.Syft
import org.openmined.syft.demo.MyApp
import org.openmined.syft.demo.federated.domain.MNISTDataRepository
import org.openmined.syft.demo.federated.domain.SyntheticDataRepository
import org.openmined.syft.demo.federated.domain.FEMNISTDataRepository
import org.openmined.syft.demo.federated.domain.TrainingTask
import org.openmined.syft.demo.federated.logging.MnistLogger
import org.openmined.syft.demo.federated.service.EPOCH
import org.openmined.syft.demo.federated.service.LOG
import org.openmined.syft.demo.federated.service.LOSS_LIST
import org.openmined.syft.demo.federated.service.STATUS
import org.openmined.syft.demo.federated.service.WorkerRepository
import org.openmined.syft.demo.federated.ui.ContentState
import org.openmined.syft.demo.federated.ui.ProcessData
import org.openmined.syft.demo.utils.MODEL_NAME
import org.openmined.syft.demo.utils.MODEL_VERSION
import org.openmined.syft.demo.utils.getAuthToken
import org.openmined.syft.demo.utils.setTrainingMode
import org.openmined.syft.demo.utils.showMessage
import org.openmined.syft.domain.SyftConfiguration
import org.openmined.syft.execution.JobStatusSubscriber
import org.openmined.syft.execution.Plan
import org.openmined.syft.networking.datamodels.ClientConfig
import org.openmined.syft.networking.datamodels.syft.TrainingStatsRequest
import org.openmined.syft.proto.SyftModel
import timber.log.Timber
import java.net.UnknownHostException
import java.util.concurrent.ConcurrentHashMap

class MnistActivityViewModel(
    val baseUrl: String,
    private val authToken: String,
    private val workerRepository: WorkerRepository
) : MnistLogger, ViewModel() {
    override val logText
        get() = logTextInternal
    private val logTextInternal = MutableLiveData<String>()

    override val cycleText
        get() = cycleTextInternal
    private val cycleTextInternal = MutableLiveData<String>()

    override val steps
        get() = stepsInternal
    private val stepsInternal = MutableLiveData<String>()

    override val processState
        get() = processStateInternal
    private val processStateInternal = MutableLiveData<ContentState>()

    override val processData
        get() = processDataInternal
    private val processDataInternal = MutableLiveData<ProcessData>()


    private val compositeDisposable = CompositeDisposable()
    private var trainingTask: TrainingTask? = null

    override fun postState(status: ContentState) {
        processStateInternal.postValue(status)
    }

    override fun postData(result: Float) {
        processDataInternal.postValue(
            ProcessData(
                (processDataInternal.value?.data ?: emptyList()) + result
            )
        )
    }

    override fun postEpoch(epoch: String) {
        stepsInternal.postValue("Step : $epoch")
    }

    override fun postLog(message: String) {
        logTextInternal.postValue("${logTextInternal.value ?: ""}\n\n$message")
    }

    override fun postCycle(cycleId: String) {
        cycleTextInternal.postValue(cycleId)
    }

    fun launchForegroundTrainer(activity: Activity, config: SyftConfiguration, dataRepository: MNISTDataRepository, dataset: DATASET) {
        trainingTask = TrainingTask(
            config,
            authToken,
            dataRepository,
            dataset
        )
        compositeDisposable.add(trainingTask!!.runTask(this).subscribe())
    }

    fun launchForegroundTrainer(activity: Activity, config: SyftConfiguration, syntheticDataRepository: SyntheticDataRepository, dataset: DATASET) {
        trainingTask = TrainingTask(
            config,
            authToken,
            syntheticDataRepository,
            dataset
        )
        compositeDisposable.add(trainingTask!!.runTask(this).subscribe())
    }

    fun launchForegroundTrainer(activity: Activity, config: SyftConfiguration, femnistDataRepository: FEMNISTDataRepository, dataset: DATASET) {
        if(trainingTask == null) {
            trainingTask = TrainingTask(
                config,
                authToken,
                femnistDataRepository,
                dataset
            )
        }
        compositeDisposable.clear()
        compositeDisposable.add(trainingTask!!.runTask(this).subscribe())
    }

    fun disposeTraining() {
        compositeDisposable.clear()
        trainingTask?.disposeTraining()
    }

    fun getRunningWorkInfo() = workerRepository.getRunningWorkStatus()?.let {
        workerRepository.getWorkInfo(it)
    }


    fun submitJob(): LiveData<WorkInfo> {
        val requestId = workerRepository.getRunningWorkStatus()
                        ?: workerRepository.submitJob(authToken, baseUrl)
        return workerRepository.getWorkInfo(requestId)
    }

    fun cancelAllJobs() {
        workerRepository.cancelAllWork()
    }

    fun getWorkInfoObserver() = Observer { workInfo: WorkInfo? ->
        if (workInfo != null) {
            val progress = workInfo.progress
            progress.getFloat(LOSS_LIST, -2.0f).takeIf { it > -1 }?.let {
                postData(it)
            }
//            progress.getString(EPOCH, -2).takeIf { it > -1 }?.let {
//                postEpoch(it)
//            }
            progress.getString(LOG)?.let {
                postLog(it)
            }
            postState(
                ContentState.getObjectFromString(
                    progress.getString(STATUS)
                ) ?: ContentState.Training
            )
        }
    }

    fun sendWorkerStatusToServer(config: SyftConfiguration) {
        MyApp.getInstance().getDeviceToken {deviceToken ->
            var syftWorker = Syft.getInstance(config, getAuthToken())
            val token = deviceToken
            Timber.d("================== token $token")
            syftWorker.executeWorkerOnlineStatus({
                it?.let {
                    when (it) {
                        is ErrnoException, is UnknownError, is UnknownHostException, is NetworkErrorException -> {
                            showMessage("Please connect with internet")
                        }
                    }
                }

            }, MODEL_NAME, MODEL_VERSION, token)
        }

    }

    fun uploadTrainingStats(callback: (Throwable?) -> kotlin.Unit,
                            config: SyftConfiguration,
                            trainingStats: ArrayList<TrainingStatsRequest>) {
        var syftWorker = Syft.getInstance(config, getAuthToken())
        syftWorker.uploadTrainingStats({
            it?.let {
                when (it) {
                    is ErrnoException, is UnknownError, is UnknownHostException, is NetworkErrorException -> {
                        showMessage("Please connect with internet")
                    }
                    else -> {
                        callback(null)
                    }
                }
            }

        }, trainingStats)
    }

}
