package org.openmined.syft.networking.datamodels.syft

import android.util.Log
import kotlinx.serialization.*
import kotlinx.serialization.internal.SerialClassDescImpl
import kotlinx.serialization.json.*
import org.openmined.syft.networking.datamodels.NetworkModels

internal const val REPORT_STAT_TYPE = "model-centric/report_stats"

@Serializable
internal data class ReportStatResponse(
        val status: String? = null,
        val error: String? = null
) : NetworkModels()

//internal data class ReportResponseData(val type: String? = "", val data: ReportResponse)

@Serializable
internal data class Accuracy(
        @SerialName("test_accuracy")
        val testAccuracy: Float,
        @SerialName("num_samples")
        val numSamples: Int


)

@Serializable
internal data class ReportStatRequest(
        @SerialName("worker_id")
        val workerId: String,
        @SerialName("request_key")
        val requestKey: String,


        @SerialName("test_accuracy_list")
        val accuracies: ArrayList<Accuracy>,

        @SerialName("model_size")
        val modelSize: Long,

        @SerialName("avg_epoch_train_time")
        val avgEpochTrainTime: Long,

        @SerialName("total_train_time")
        val totalTrainTime: Long,
        @SerialName("cycle_id")
        val cycleId: String
) : NetworkModels() {
}
@Serializable(with = ReportStatResponseSerializer::class)
internal open class ReportStatResponseData : NetworkModels() {
        @Serializable
        data class ReportSuccess(
                val status: String? = null
        ) : ReportStatResponseData()

        @Serializable
        data class ReportError(
                val error: String? = null
        ) : ReportStatResponseData()
}

@Serializer(forClass = ReportStatResponseData::class)
internal class ReportStatResponseSerializer : KSerializer<ReportStatResponseData> {
        //    var jsonConfig = JsonConfiguration()
//    jsonConfig.str
        private val json = Json(JsonConfiguration(strictMode = false))
        override val descriptor: SerialDescriptor
                get() = SerialClassDescImpl("AuthResponseSerializer")

        override fun deserialize(decoder: Decoder): ReportStatResponseData {
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
                                ReportStatResponseData.ReportSuccess.serializer(),
                                response.getObject("data").toString()
                        )
                else
                        json.parse(ReportStatResponseData.ReportError.serializer(), data.toString())
        }

        override fun serialize(encoder: Encoder, obj: ReportStatResponseData) {
                val output = encoder as? JsonOutput
                        ?: throw SerializationException("This class can be saved only by Json")
                when (obj) {
                        is ReportStatResponseData.ReportSuccess -> {
                                val accept = json.toJson(ReportStatResponseData.ReportSuccess.serializer(),obj)
                                Log.v("ReportResponseData", "=================================>>> accept >>> " + accept)
                                output.encodeJson(accept)
                        }
                        is ReportStatResponseData.ReportError ->
                                output.encodeJson(
                                        json.toJson(
                                                ReportStatResponseData.ReportError.serializer(),
                                                obj
                                        )
                                )
                }
        }
}
