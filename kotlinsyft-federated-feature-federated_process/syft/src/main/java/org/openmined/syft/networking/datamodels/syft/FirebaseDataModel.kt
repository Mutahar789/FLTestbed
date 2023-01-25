package org.openmined.syft.networking.datamodels.syft

import android.util.Log
import kotlinx.serialization.*
import kotlinx.serialization.internal.SerialClassDescImpl
import kotlinx.serialization.json.*
import org.openmined.syft.networking.datamodels.NetworkModels

internal const val FCM_TYPE = "model-centric/save-fcm-token"

@Serializable
internal data class TokenResponse(
        val status: String? = null,
        val error: String? = null
) : NetworkModels()

//internal data class ReportResponseData(val type: String? = "", val data: ReportResponse)

@Serializable
internal data class TokenRequest(
        @SerialName("worker_id")
        val workerId: String,
        @SerialName("fcm_device_token")
        val fcmToken: String,
        val model: String,
        val version: String,
        @SerialName("ram_size")
        val ramSize: Float,
        @SerialName("cpu_cores")
        val cpuCores: Int
) : NetworkModels()

@Serializable(with = TokenResponseSerializer::class)
internal open class TokenResponseData : NetworkModels() {
        @Serializable
        data class TokenSuccess(
                val status: String? = null
        ) : TokenResponseData()

        @Serializable
        data class TokenError(
                val error: String? = null
        ) : TokenResponseData()
}

@Serializer(forClass = TokenResponseData::class)
internal class TokenResponseSerializer : KSerializer<TokenResponseData> {
        //    var jsonConfig = JsonConfiguration()
//    jsonConfig.str
        private val json = Json(JsonConfiguration(strictMode = false))
        override val descriptor: SerialDescriptor
                get() = SerialClassDescImpl("AuthResponseSerializer")

        override fun deserialize(decoder: Decoder): TokenResponseData {
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

                Log.v("TokenResponseData", "========================== >>> data  " + data)
                Log.v("TokenResponseData", "========================== >>> response  " + response)

                return if (!response.getObject("data").getPrimitive("status").content.isNullOrEmpty())
                        json.parse(
                                TokenResponseData.TokenSuccess.serializer(),
                                response.getObject("data").toString()
                        )
                else
                        json.parse(TokenResponseData.TokenError.serializer(), data.toString())
        }

        override fun serialize(encoder: Encoder, obj: TokenResponseData) {
                val output = encoder as? JsonOutput
                        ?: throw SerializationException("This class can be saved only by Json")
                when (obj) {
                        is TokenResponseData.TokenSuccess -> {
                                val accept = json.toJson(TokenResponseData.TokenSuccess.serializer(),obj)
                                Log.v("ReportResponseData", "=================================>>> accept >>> " + accept)
                                output.encodeJson(accept)
                        }
                        is TokenResponseData.TokenError ->
                                output.encodeJson(
                                        json.toJson(
                                                TokenResponseData.TokenError.serializer(),
                                                obj
                                        )
                                )
                }
        }
}
