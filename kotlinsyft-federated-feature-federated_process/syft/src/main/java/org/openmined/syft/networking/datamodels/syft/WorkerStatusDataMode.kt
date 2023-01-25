package org.openmined.syft.networking.datamodels.syft

import android.util.Log
import kotlinx.serialization.*
import kotlinx.serialization.internal.SerialClassDescImpl
import kotlinx.serialization.json.*
import org.openmined.syft.networking.datamodels.NetworkModels

internal const val WORKER_UPDATE_STATUS_TYPE = "model-centric/update-worker-status"
@Serializable
internal data class WorkerStatusRequest(
        @SerialName("worker_id")
        val workerId: String,
        @SerialName("auth_token")
        val authToken: String,
        val model: String,
        val version: String? = null,
        val fcm_push_token: String
) : NetworkModels()