package org.openmined.syft.demo.login

import androidx.lifecycle.AndroidViewModel
import org.openmined.syft.Syft
import org.openmined.syft.demo.BuildConfig.SYFT_AUTH_TOKEN
import org.openmined.syft.demo.utils.MODEL_NAME
import org.openmined.syft.demo.utils.MODEL_VERSION
import org.openmined.syft.demo.MyApp
import org.openmined.syft.domain.SyftConfiguration
import org.openmined.syft.execution.JobStatusSubscriber
import org.openmined.syft.execution.Plan
import org.openmined.syft.networking.datamodels.ClientConfig
import org.openmined.syft.proto.SyftModel
import timber.log.Timber
import java.util.concurrent.ConcurrentHashMap

class LoginViewModel(var configuration: SyftConfiguration, var application: MyApp, var baseUrl: String) : AndroidViewModel(application) {
    private var syftWorker: Syft? = null

    fun checkUrl(baseUrl: String): Boolean {
        return true
    }

    fun getAuthToken() : String = SYFT_AUTH_TOKEN

    fun authenticate(callback: (Throwable?) -> Unit, ramSize: Float, cpuCores: Int) {

        MyApp.getInstance().getDeviceToken {deviceToken ->
            syftWorker = Syft.getInstance(configuration, getAuthToken())
            val token = deviceToken
            Timber.d("================== token $token")
            val mnistJob = syftWorker!!.newJob(MODEL_NAME, MODEL_VERSION, token)

            Timber.d("Aunthentication job started \n")

            val jobStatusSubscriber = object : JobStatusSubscriber() {
                override fun onReady(
                    model: SyftModel,
                    CycleId: String,
                    plans: ConcurrentHashMap<String, Plan>,
                    clientConfig: ClientConfig
                ) {
                    Timber.d("========================================= onready")
                    Timber.d("Token has been saved on server")
//                    syftWorker?.dispose()
                    mnistJob.markJobCompleted()
                    callback(null)

//                    syftWorker?.dispose()
                }

                override fun onComplete() {
                    Timber.d("========================================= oncomplete")
                    mnistJob.markJobCompleted()
                    callback(null)
                }

                override fun onError(throwable: Throwable) {
                    Timber.d("========================================= onerror")
                    throwable.printStackTrace()
                    mnistJob.markJobCompleted()
                    callback(throwable)
                }
            }

            mnistJob.executeFirebaseTokenRequest(jobStatusSubscriber, ramSize, cpuCores)
        }
    }

}