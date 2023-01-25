package org.openmined.syft.networking.datamodels.syft

import android.util.Log
import kotlinx.serialization.*
import kotlinx.serialization.internal.SerialClassDescImpl
import kotlinx.serialization.json.*
import org.openmined.syft.networking.datamodels.NetworkModels

internal const val REPORT_TYPE = "model-centric/report"

@Serializable
internal data class ReportResponse(
        val status: String? = null,
        val error: String? = null
) : NetworkModels()

//internal data class ReportResponseData(val type: String? = "", val data: ReportResponse)

@Serializable
internal data class ReportRequest(
        @SerialName("worker_id")
        val workerId: String,
        @SerialName("request_key")
        val requestKey: String,
        val diff: String,

        @SerialName("test_accuracy_list")
        val accuracies: ArrayList<Accuracy>,

        val train_num_samples: Int,

        @SerialName("model_size")
        val modelSize: Long,

        @SerialName("avg_epoch_train_time")
        val avgEpochTrainTime: Long,

        @SerialName("total_train_time")
        val totalTrainTime: Long,

        @SerialName("cycle_id")
        val cycleId: String,

        @SerialName("trained_epochs")
        val trained_epochs: Int,

        @SerialName("model_download_time")
        val modelDownloadTime: String,

        @SerialName("model_report_time")
        val modelReportTime: String,

        @SerialName("model_download_size")
        val modelDownloadSize: String,

        @SerialName("model_report_size")
        val modelReportSize: String,

        val avgActivations: String? = null
) : NetworkModels() {
}
@Serializable(with = ReportResponseSerializer::class)
internal open class ReportResponseData : NetworkModels() {
        @Serializable
        data class ReportSuccess(
                val status: String? = null
        ) : ReportResponseData()

        @Serializable
        data class ReportError(
                val error: String? = null
        ) : ReportResponseData()
}

@Serializer(forClass = ReportResponseData::class)
internal class ReportResponseSerializer : KSerializer<ReportResponseData> {
        //    var jsonConfig = JsonConfiguration()
//    jsonConfig.str
        private val json = Json(JsonConfiguration(strictMode = false))
        override val descriptor: SerialDescriptor
                get() = SerialClassDescImpl("AuthResponseSerializer")

        override fun deserialize(decoder: Decoder): ReportResponseData {
                val input = decoder as? JsonInput
                        ?: throw SerializationException("This class can be loaded only by Json")
                val response = input.decodeJson() as? JsonObject
                        ?: throw SerializationException("Expected JsonObject")
                val data = json {
                        response.forEach { key, value ->
                                if (key != "status")
                                        key to value
                        }
                }

                Log.v("ReportResponseData", "========================== >>> data  " + data)
                Log.v("ReportResponseData", "========================== >>> response  " + response)

                return if (!response.getObject("data").getPrimitive("status").content.isNullOrEmpty())
                        json.parse(
                                ReportResponseData.ReportSuccess.serializer(),
                                response.getObject("data").toString()
                        )
                else
                        json.parse(ReportResponseData.ReportError.serializer(), data.toString())
        }

        override fun serialize(encoder: Encoder, obj: ReportResponseData) {
                val output = encoder as? JsonOutput
                        ?: throw SerializationException("This class can be saved only by Json")
                when (obj) {
                        is ReportResponseData.ReportSuccess -> {
                                val accept = json.toJson(ReportResponseData.ReportSuccess.serializer(),obj)
                                Log.v("ReportResponseData", "=================================>>> accept >>> " + accept)
                                output.encodeJson(accept)
                        }
                        is ReportResponseData.ReportError ->
                                output.encodeJson(
                                        json.toJson(
                                                ReportResponseData.ReportError.serializer(),
                                                obj
                                        )
                                )
                }
        }
}
