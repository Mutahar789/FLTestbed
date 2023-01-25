package org.openmined.syft.networking.datamodels.syft

import android.util.Log
import kotlinx.serialization.*
import kotlinx.serialization.internal.SerialClassDescImpl
import kotlinx.serialization.json.*
import org.openmined.syft.networking.datamodels.NetworkModels

internal const val TRAINING_STATS_STATUS_TYPE = "model-centric/upload_stats"
@Serializable
data class TrainingStatsRequest(
        @SerialName("worker_id")
        val workerId: String,

        @SerialName("cycle_id")
        val cycleId: Int,

        val accuracy: Float,

        @SerialName("num_samples")
        val numSamples: Int

) : NetworkModels()
@Serializable
data class TrainingStatsRequestList(
        val stats: ArrayList<TrainingStatsRequest>

) : NetworkModels()

