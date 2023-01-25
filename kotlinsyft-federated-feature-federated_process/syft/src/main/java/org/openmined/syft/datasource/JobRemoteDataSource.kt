package org.openmined.syft.datasource

import android.util.Log
import io.reactivex.Single
import org.openmined.syft.networking.requests.HttpAPI
import org.openmined.syft.proto.SyftModel
import java.io.InputStream

internal class JobRemoteDataSource(private val httpApi: HttpAPI) {

    /**
     * Downloads the model modelId and offers it as an input stream for further processing.
     * @param workerId
     * @param requestKey
     * @param modelId
     */
    fun downloadModel(workerId: String, requestKey: String, modelId: String): Single<InputStream> {
        return httpApi.downloadModel(workerId, requestKey, modelId)

                .map {
                    it.body()?.byteStream()
                }
    }

    fun downloadModel(model: SyftModel, workerId: String, requestKey: String, modelId: String): Single<InputStream> {
//        model.setModelDownloadStartTime()
        return httpApi.downloadModel(workerId, requestKey, modelId)
            .map {
                Log.e("it.raw()", "it.raw().receivedResponseAtMillis ${it.raw().receivedResponseAtMillis}, it.raw().sentRequestAtMillis ${it.raw().sentRequestAtMillis}")
                it.body()?.byteStream()

            }
    }

    fun downloadProtocol(
        workerId: String,
        requestKey: String,
        protocolId: String
    ): Single<InputStream> {
        return httpApi.downloadProtocol(workerId, requestKey, protocolId)
                .map {
                    it.body()?.byteStream()
                }
    }

    fun downloadPlan(
        workerId: String,
        requestKey: String,
        planId: String,
        opType: String
    ): Single<InputStream> {
        return httpApi.downloadPlan(workerId, requestKey, planId, opType)
                .map {
                    it.body()?.byteStream()
                }
    }
}