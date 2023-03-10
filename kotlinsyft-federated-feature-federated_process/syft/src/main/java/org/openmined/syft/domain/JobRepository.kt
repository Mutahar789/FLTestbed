package org.openmined.syft.domain

import android.util.Log
import io.reactivex.Single
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.processors.PublishProcessor
import org.openmined.syft.datasource.JobLocalDataSource
import org.openmined.syft.datasource.JobRemoteDataSource
import org.openmined.syft.execution.JobStatusMessage
import org.openmined.syft.execution.Plan
import org.openmined.syft.execution.Protocol
import org.openmined.syft.networking.datamodels.ClientConfig
import org.openmined.syft.proto.SyftModel
import java.io.InputStream
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicReference

internal const val PLAN_OP_TYPE = "torchscript"
private const val TAG = "JobDownloader"

@ExperimentalUnsignedTypes
internal class JobRepository(
    private val jobLocalDataSource: JobLocalDataSource,
    private val jobRemoteDataSource: JobRemoteDataSource
) {

    private val trainingParamsStatus = AtomicReference(DownloadStatus.NOT_STARTED)
    val status: DownloadStatus
        get() = trainingParamsStatus.get()

    fun getDiffScript(config: SyftConfiguration) =
            jobLocalDataSource.getDiffScript(config)

    fun persistToLocalStorage(
        input: InputStream,
        parentDir: String,
        fileName: String,
        overwrite: Boolean = false
    ): String {
        return jobLocalDataSource.save(input, parentDir, fileName, overwrite)
    }

    fun downloadData(
        workerId: String,
        config: SyftConfiguration,
        requestKey: String,
        cycleId: String,
        networkDisposable: CompositeDisposable,
        jobStatusProcessor: PublishProcessor<JobStatusMessage>,
        clientConfig: ClientConfig?,
        plans: ConcurrentHashMap<String, Plan>,
        model: SyftModel,
        protocols: ConcurrentHashMap<String, Protocol>
    ) {
        Log.d(TAG, "beginning download")
        trainingParamsStatus.set(DownloadStatus.RUNNING)

        networkDisposable.add(
            Single.zip(
                getDownloadables(
                    workerId,
                    config,
                    requestKey,
                    model,
                    plans,
                    protocols,
                    cycleId
                )
            ) { successMessages ->
                successMessages.joinToString(
                    ",",
                    prefix = "files ",
                    postfix = " downloaded successfully"
                )
            }
                    .compose(config.networkingSchedulers.applySingleSchedulers())
                    .subscribe(
                        { successMsg: String ->
                            Log.d(TAG, successMsg)
                            trainingParamsStatus.set(DownloadStatus.COMPLETE)
                            jobStatusProcessor.offer(
                                JobStatusMessage.JobReady(
                                    model,
                                    cycleId,
                                    plans,
                                    clientConfig
                                )
                            )
                        },
                        { e ->
                            jobStatusProcessor.onError(e)
                        }
                    )
        )
    }

    private fun getDownloadables(
        workerId: String,
        config: SyftConfiguration,
        request: String,
        model: SyftModel,
        plans: ConcurrentHashMap<String, Plan>,
        protocols: ConcurrentHashMap<String, Protocol>,
        cycleId: String
    ): List<Single<String>> {
        val downloadList = mutableListOf<Single<String>>()
        plans.forEach { (_, plan) ->
            downloadList.add(
                processPlans(
                    workerId,
                    config,
                    request,
                    "${config.filesDir}/plans",
                    plan
                )
            )
        }
        protocols.forEach { (_, protocol) ->
            protocol.protocolFileLocation = "${config.filesDir}/protocols"
            downloadList.add(
                processProtocols(
                    workerId,
                    config,
                    request,
                    protocol.protocolFileLocation,
                    protocol.protocolId
                )
            )
        }
        downloadList.add(processModel(workerId, config, request, model))
        return downloadList
    }

    private fun processModel(
        workerId: String,
        config: SyftConfiguration,
        requestKey: String,
        model: SyftModel
    ): Single<String> {
        val modelId = model.pyGridModelId ?: throw IllegalStateException("Model id not initiated")
//        model.setModelDownloadStartTime()
        return jobRemoteDataSource.downloadModel(model, workerId, requestKey, modelId)
                .flatMap { modelInputStream ->
                    jobLocalDataSource.saveAsync(
                        modelInputStream,
                        "${config.filesDir}/models",
                        "$modelId.pb"
                    )
                }.flatMap { modelFile ->
                    Single.create<String> { emitter ->
                        model.loadModelState(modelFile)
                        emitter.onSuccess(modelFile)
                    }
                }
                .compose(config.networkingSchedulers.applySingleSchedulers())
    }

    private var planMap: HashMap<String, Boolean> = hashMapOf()

    private fun processPlans(
        workerId: String,
        config: SyftConfiguration,
        requestKey: String,
        destinationDir: String,
        plan: Plan
    ): Single<String> {
//        if(planMap.size > 0)
//        planMap.clear()
        return jobRemoteDataSource.downloadPlan(
            workerId,
            requestKey,
            plan.planId,
            PLAN_OP_TYPE
        )
                .flatMap { planInputStream ->
                    jobLocalDataSource.saveAsync(
                        planInputStream,
                        destinationDir,
                        "${plan.planId}.pb"
                    )
                }.flatMap { filepath ->
                    Single.create<String> { emitter ->
                        val torchscriptLocation = jobLocalDataSource.saveTorchScript(
                            destinationDir,
                            filepath,
                            "torchscript_${plan.planId}.pt"
                        )
//                        Log.e("JobRepository", torchscriptLocation)
//                        Log.e("JobRepository", "planMap $planMap")
                        if(!planMap.containsKey(torchscriptLocation)) {
                            planMap.put(torchscriptLocation, true)
                            plan.loadScriptModule(torchscriptLocation)
                        }
//                        plan.loadScriptModule(torchscriptLocation)
                        emitter.onSuccess(filepath)
                    }
                }
                .compose(config.networkingSchedulers.applySingleSchedulers())

    }

    private fun processProtocols(
        workerId: String,
        config: SyftConfiguration,
        requestKey: String,
        destinationDir: String,
        protocolId: String
    ): Single<String> {
        return jobRemoteDataSource.downloadProtocol(workerId, requestKey, protocolId)
                .flatMap { protocolInputStream ->
                    jobLocalDataSource.saveAsync(
                        protocolInputStream,
                        destinationDir,
                        "$protocolId.pb"
                    )
                }
                .compose(config.networkingSchedulers.applySingleSchedulers())
    }
}

enum class DownloadStatus {
    NOT_STARTED, RUNNING, COMPLETE
}
