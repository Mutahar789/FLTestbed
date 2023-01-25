package org.openmined.syft

import android.util.Log
import io.reactivex.Scheduler
import io.reactivex.Single
import io.reactivex.android.schedulers.AndroidSchedulers
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.disposables.Disposable
import kotlinx.serialization.SerialName
import org.openmined.syft.domain.SyftConfiguration
import org.openmined.syft.execution.JobErrorThrowable
import org.openmined.syft.execution.JobStatusSubscriber
import org.openmined.syft.execution.SyftJob
import org.openmined.syft.fp.Either
import org.openmined.syft.monitor.DeviceMonitor
import org.openmined.syft.networking.datamodels.syft.*
import org.openmined.syft.networking.datamodels.syft.CycleResponseData
import org.openmined.syft.networking.datamodels.syft.ReportResponseData
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "Syft"

/**
 * This is the main syft worker handling creation and deletion of jobs. This class is also responsible for monitoring device resources via DeviceMonitor
 */
@ExperimentalUnsignedTypes
class Syft internal constructor(
    private val syftConfig: SyftConfiguration,
    private val deviceMonitor: DeviceMonitor,
    private val authToken: String?
) : Disposable {
    companion object {
        @Volatile
        private var INSTANCE: Syft? = null
        public var plansMap: HashMap<String, Boolean> = hashMapOf()

        /**
         * Only a single worker must be instantiated across an app lifecycle.
         * The [getInstance] ensures creation of the singleton object if needed or returns the already created worker.
         * This method is thread safe so getInstance calls across threads do not suffer
         * @param syftConfiguration The SyftConfiguration object specifying the mutable properties of syft worker
         * @param authToken (Optional) The JWT token to be passed by the user
         * @return Syft instance
         */
        fun getInstance(
            syftConfiguration: SyftConfiguration,
            authToken: String? = null
        ): Syft {
            return INSTANCE ?: synchronized(this) {
                INSTANCE?.let {
                    if (it.syftConfig == syftConfiguration && it.authToken == authToken) it
                    else throw ExceptionInInitializerError("syft worker initialised with different parameters. Dispose previous worker")
                } ?: Syft(
                    syftConfig = syftConfiguration,
                    deviceMonitor = DeviceMonitor.construct(syftConfiguration),
                    authToken = authToken
                ).also { INSTANCE = it }
            }
        }
    }

    //todo single job for now but eventually worker should support multiple jobs
    private var workerJob: SyftJob? = null
    private val compositeDisposable = CompositeDisposable()
    private val isDisposed = AtomicBoolean(false)

    @Volatile
    private var workerId: String? = null

    /**
     * Create a new job for the worker.
     * @param model specifies the model name by which the parameters are hosted on the PyGrid server
     * @param version the version of the model on PyGrid
     * @return [SyftJob]
     */
    fun newJob(
            model: String,
            version: String? = null,
            deviceToken: String? = null
    ): SyftJob {
        val job = SyftJob.create(
                model,
                version,
                deviceToken,
                this,
                syftConfig
        )
        if (workerJob != null)
//            return job
            throw IndexOutOfBoundsException("maximum number of allowed jobs reached")

        workerJob = job
        job.subscribe(object : JobStatusSubscriber() {
            override fun onComplete() {
                workerJob = null
            }

            override fun onError(throwable: Throwable) {
                Log.e(TAG, throwable.message.toString())
                workerJob = null
            }
        }, syftConfig.networkingSchedulers)

        return job
    }



    fun newJob(
        model: String,
        version: String? = null,
        deviceToken: String? = null,
        cycleStartRequestKey: String
    ): SyftJob {
        val job = SyftJob.create(
            model,
            version,
            deviceToken,
            cycleStartRequestKey,
            this,
            syftConfig
        )
        if (workerJob != null)
            throw IndexOutOfBoundsException("maximum number of allowed jobs reached")

        workerJob = job
        job.subscribe(object : JobStatusSubscriber() {
            override fun onComplete() {
                workerJob = null
            }

            override fun onError(throwable: Throwable) {
                Log.e(TAG, throwable.message.toString())
                workerJob = null
            }
        }, syftConfig.networkingSchedulers)

        return job
    }

    fun getSyftWorkerId() = workerId

    internal fun executeCycleRequest(job: SyftJob, fcmToken: String="", ramSize: Float=0f, cores: Int=0) {
        if (job.throwErrorIfBatteryInvalid() || job.throwErrorIfNetworkInvalid())
            return
        Log.v("Syft", "========================>>>>>>>>>>>>>>>>>>>> executeCycleRequest $workerId, cycleStartPushKey: ${job.getCycleStartRequestKey()}")
        workerId?.let { id ->
            compositeDisposable.add(
                deviceMonitor.getNetworkStatus(id, job.requiresSpeedTest.get())
                        .flatMap { networkState ->
                            requestCycle(
                                id,
                                job,
                                networkState.ping,
                                networkState.downloadSpeed,
                                networkState.uploadSpeed,
                                job.getCycleStartRequestKey(),
                                fcmToken,
                                ramSize,
                                cores
                            )
                        }
                        .compose(syftConfig.networkingSchedulers.applySingleSchedulers())
                        .subscribe(
                            { response: CycleResponseData ->
                                when (response) {
                                    is CycleResponseData.CycleAccept -> handleCycleAccept(response)
                                    is CycleResponseData.CycleReject -> handleCycleReject(response)
                                }
                            },
                            { errorMsg: Throwable ->
                                job.publishError(
                                    JobErrorThrowable.ExternalException(
                                        errorMsg.message,
                                        errorMsg.cause
                                    )
                                )
                            })
            )
        } ?: executeAuthentication(job)
    }

    internal fun executeFirebaseTokenRequest(job: SyftJob, ramSize: Float, cpuCores: Int) {
        if (job.throwErrorIfBatteryInvalid() || job.throwErrorIfNetworkInvalid())
            return

        Log.v("Syft", "========================>>>>>>>>>>>>>>>>>>>> executeFirebaseTokenRequest $workerId")
        workerId?.let { id ->
            compositeDisposable.add(
                    saveFirebaseToken(id,job, ramSize, cpuCores)
                            .compose(syftConfig.networkingSchedulers.applySingleSchedulers())
                            .subscribe(
                                    { response: TokenResponseData ->
                                        when(response) {
                                            is TokenResponseData.TokenError -> job.publishError(JobErrorThrowable.NetworkResponseFailure(response.error!!))
                                            is TokenResponseData.TokenSuccess -> {
                                                Log.d("Syft", "Firebase Token status status ${response.status!!}")
                                                job.markJobCompleted()
                                            }
                                        }
                                    },
                                    { errorMsg: Throwable ->
                                        job.publishError(
                                                JobErrorThrowable.ExternalException(
                                                        errorMsg.message,
                                                        errorMsg.cause
                                                )
                                        )
                                    })
            )
        } ?: executeAuthentication({}, job, ramSize, cpuCores)
    }

    fun executeWorkerOnlineStatus(callback: (Throwable?) -> kotlin.Unit, modelName: String, modelVersion: String, deviceToken: String) {
        Log.d("Syft", "executeWorkerOnlineStatus workerId $workerId")
        workerId?.let { id ->
            compositeDisposable.add(
                updateWorkerOnlineStatus(id,modelName, modelVersion, deviceToken)
                        .compose(syftConfig.networkingSchedulers.applySingleSchedulers())
                        .subscribe(
                            { response: TokenResponseData ->
                                when(response) {
                                    is TokenResponseData.TokenError -> /*job.publishError(JobErrorThrowable.NetworkResponseFailure(response.error!!))*/ callback(null)
                                    is TokenResponseData.TokenSuccess -> {
                                        Log.d("Syft", "executeWorkerOnlineStatus status ${response.status!!}")
                                        callback(null)
                                    }
                                }
                            },
                            { errorMsg: Throwable ->
                                callback(errorMsg)
                            })
            )
        }
    }

    fun uploadTrainingStats(callback: (Throwable?) -> kotlin.Unit,
                            trainingStats: ArrayList<TrainingStatsRequest>) {
        workerId?.let { id ->
            compositeDisposable.add(
                uploadTrainingStats(trainingStats)
                        .compose(syftConfig.networkingSchedulers.applySingleSchedulers())
                        .subscribe(
                            { response: TokenResponseData ->
                                when(response) {
                                    is TokenResponseData.TokenError -> callback(null)
                                    is TokenResponseData.TokenSuccess -> {
                                        Log.d("Syft", "executeWorkerOnlineStatus status ${response.status!!}")
                                        callback(null)
                                    }
                                }
                            },
                            { errorMsg: Throwable ->
                                callback(errorMsg)
                            })
            )
        }
    }

    /**
     * Check if the syft worker has been disposed
     * @return True/False
     */
    override fun isDisposed() = isDisposed.get()

    /**
     * Explicitly dispose off the worker. All the jobs running in the worker will be disposed off as well.
     * Clears the current singleton worker instance so the immediately next [getInstance] call creates a new syft worker
     */
    override fun dispose() {
        Log.d(TAG, "disposing syft worker")
        deviceMonitor.dispose()
        compositeDisposable.clear()
        workerJob?.dispose()
        INSTANCE = null
    }

    internal fun isNetworkValid() = true//deviceMonitor.isNetworkStateValid()
    internal fun isBatteryValid() = true//deviceMonitor.isBatteryStateValid()

    private fun requestCycle(
            id: String,
            job: SyftJob,
            pingd: Int?,
            downldoadSpeed: Float?,
            uploaddSpeed: Float?,
    cycleStartRequestKey: String,
            fcmToken:String,
            ramSize:Float,
            cores:Int
    ): Single<CycleResponseData> {
        val downloadSpeed = 20000.toFloat()
        val uploadSpeed = 20000.toFloat()
        val ping = 1
        Log.e("Syft", "Download speed: $downloadSpeed, upload: $uploadSpeed")
        return when (val check = checkConditions(ping, downloadSpeed, uploadSpeed)) {
            is Either.Left -> Single.error(JobErrorThrowable.NetworkUnreachable(check.a))
            is Either.Right -> syftConfig.getSignallingClient().getCycle(
                CycleRequest(
                    id,
                    job.jobId.modelName,
                    job.jobId.version,
                    ping ?: -1,
                    downloadSpeed ?: 0.0f,
                    uploadSpeed ?: 0.0f,
                    cycleStartRequestKey,
                    fcmToken,
                    ramSize,
                    cores
                )
            )
        }
    }

    private fun saveFirebaseToken(
            workerId: String,
            job: SyftJob,
            ramSize: Float,
            cpuCores: Int
    ): Single<TokenResponseData> {
        return syftConfig.getSignallingClient().saveFirebaseDeviceToken(
                TokenRequest(workerId,job.deviceToken!!, job.modelName, job.version!!, ramSize, cpuCores)
        )
    }

    private fun updateWorkerOnlineStatus(
        workerId: String,
        modelName: String,
        modelVersion: String,
        deviceToken: String
    ): Single<TokenResponseData> {
        return syftConfig.getSignallingClient().updateWorkerOnlineStatus(
            WorkerStatusRequest(workerId, authToken!!, modelName, modelVersion, deviceToken)
        )
    }

    private fun uploadTrainingStats(
        trainingStats: ArrayList<TrainingStatsRequest>
    ): Single<TokenResponseData> {
        return syftConfig.getSignallingClient().uploadMetrics(TrainingStatsRequestList(trainingStats)
        )
    }

    private fun checkConditions(
        ping: Int?,
        downloadSpeed: Float?,
        uploadSpeed: Float?
    ): Either<String, Boolean> {
        return when {
            ping == null ->
                Either.Left("unable to get ping")
            downloadSpeed == null ->
                Either.Left("unable to verify download speed")
            uploadSpeed == null ->
                Either.Left("unable to verify upload speed")
            else -> Either.Right(true)
        }
    }

    private fun handleCycleReject(responseData: CycleResponseData.CycleReject) {
        workerJob?.cycleRejected(responseData)
    }

    private fun handleCycleAccept(responseData: CycleResponseData.CycleAccept) {
        val job = workerJob ?: throw IllegalStateException("job deleted and accessed")
        job.cycleAccepted(responseData)
        if (job.throwErrorIfBatteryInvalid() ||
            job.throwErrorIfNetworkInvalid()
        )
            return

        workerId?.let {
            job.downloadData(it, responseData)
        } ?: job.publishError(JobErrorThrowable.UninitializedWorkerError)

    }

    private fun executeAuthentication(job: SyftJob) {
        compositeDisposable.add(
            syftConfig.getSignallingClient().authenticate(
                AuthenticationRequest(
                    authToken,
                    job.jobId.modelName,
                    job.jobId.version
                )
            )
                    .compose(syftConfig.networkingSchedulers.applySingleSchedulers())
                    .subscribe({ response: AuthenticationResponse ->
                        when (response) {
                            is AuthenticationResponse.AuthenticationSuccess -> {
                                if (workerId == null) {
                                    setSyftWorkerId(response.workerId)
                                }
                                //todo eventually requires_speed test will be migrated to it's own endpoint
                                job.requiresSpeedTest.set(response.requiresSpeedTest)
                                executeCycleRequest(job)
                            }
                            is AuthenticationResponse.AuthenticationError -> {
                                job.publishError(JobErrorThrowable.AuthenticationFailure(response.errorMessage))
                                Log.d(TAG, response.errorMessage)
                            }
                        }
                    }, {
                        job.publishError(JobErrorThrowable.ExternalException(it.message, it.cause))
                    })
        )
    }

    private fun executeAuthentication(callback: () -> Unit, job: SyftJob, ramSize: Float, cpuCores: Int) {
        Log.v("Syft", "========================>>>>>>>>>>>>>>>>>>>> executeAuthentication")
        compositeDisposable.add(
                syftConfig.getSignallingClient().authenticate( AuthenticationRequest(authToken,job.jobId.modelName,job.jobId.version))
                        .compose(syftConfig.networkingSchedulers.applySingleSchedulers())
                        .subscribe({ response: AuthenticationResponse ->
                            when (response) {
                                is AuthenticationResponse.AuthenticationSuccess -> {
                                    if (workerId == null) {
                                        setSyftWorkerId(response.workerId)
                                    }
                                    //todo eventually requires_speed test will be migrated to it's own endpoint
                                    executeFirebaseTokenRequest(job, ramSize, cpuCores)
                                    Log.v("Syft", "========================>>>>>>>>>>>>>>>>>>>> executeAuthentication response: $workerId")
                                }
                                is AuthenticationResponse.AuthenticationError -> {
                                    job.publishError(JobErrorThrowable.AuthenticationFailure(response.errorMessage))
                                    Log.d(TAG, response.errorMessage)
                                }
                            }
                        }, {
                            job.publishError(JobErrorThrowable.ExternalException(it.message, it.cause))
                        })
        )
    }

    @Synchronized
    fun setSyftWorkerId(workerId: String) {
        Log.v("Syft", "============================= setSyftWorkerId")
        if (this.workerId == null)
            this.workerId = workerId
        else if (workerJob == null)
            this.workerId = workerId
    }

    private fun disposeSocketClient() {
        syftConfig.getWebRTCSignallingClient().dispose()
    }

}
