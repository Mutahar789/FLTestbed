package org.openmined.syft.networking.datamodels.webRTC

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import org.openmined.syft.networking.datamodels.NetworkModels

@Serializable
internal data class JoinRoomRequest(
    @SerialName("worker_id")
    val workerId: String,
    @SerialName("scope_id")
    val scopeId: String
) : NetworkModels()

@Serializable
internal data class JoinRoomResponse(
    @SerialName("worker_id")
    val workerId: String,
    @SerialName("scope_id")
    val scopeId: String
) : NetworkModels()