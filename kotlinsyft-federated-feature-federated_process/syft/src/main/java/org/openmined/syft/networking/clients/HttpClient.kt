package org.openmined.syft.networking.clients

import android.util.Log
import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import kotlinx.serialization.json.Json
import okhttp3.Call
import okhttp3.EventListener
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import org.openmined.syft.networking.requests.HttpAPI
import org.openmined.syft.networking.requests.NetworkingProtocol
import retrofit2.Retrofit
import retrofit2.adapter.rxjava2.RxJava2CallAdapterFactory
import java.util.concurrent.TimeUnit


/**
 * @property apiClient the retrofit api client
 */
internal class HttpClient(val apiClient: HttpAPI) {

    companion object {
        private lateinit var measurementCallback: (String, String) -> Unit

        /**
         * Creates a retrofit api client for PyGrid endpoints.
         *
         * @param baseUrl url of the server hosting the pyGrid instance.
         * @see org.openmined.syft.networking.requests.HttpAPI for endpoint description.
         */
        fun initialize(baseUrl: String): HttpClient {
            var logging = HttpLoggingInterceptor()
            logging.level = HttpLoggingInterceptor.Level.BODY
            val okHttpClientBuilder = OkHttpClient.Builder()
                .connectTimeout(5, TimeUnit.MINUTES) // connect timeout
                .writeTimeout(5, TimeUnit.MINUTES) // write timeout
                .readTimeout(5, TimeUnit.MINUTES)

                .eventListener(object : EventListener() {
//                    override fun callStart(call: Call) {
//                        val bytesEnd = System.currentTimeMillis()
//                        if(call.request().url.toString().contains("get-model")) {
//                            measurementCallback("download_time_start", "$bytesEnd")
//                        } else if(call.request().url.toString().contains("report")) {
//                            measurementCallback("report_time_start", "$bytesEnd")
//                        }
//                    }
//
//                    override fun callEnd(call: Call) {
//                        val bytesEnd = System.currentTimeMillis()
//                        if(call.request().url.toString().contains("report")) {
//                            measurementCallback("report_time_end", "$bytesEnd")
//                        } else if(call.request().url.toString().contains("get-model")) {
//                            measurementCallback("download_time_end", "$bytesEnd")
//                        }
//                    }
                    override fun responseBodyEnd(call: Call, byteCount: Long) {
                        val bytesEnd = System.currentTimeMillis()

                        if(call.request().url.toString().contains("report")) {
                            measurementCallback("report_time_end", "$bytesEnd")
                        } else if(call.request().url.toString().contains("get-model")) {
                            measurementCallback("download_time_end", "$bytesEnd")
                            measurementCallback("model_download_size", "${(byteCount / 1024) / 1024}")
                        }


                    }

//                    override fun responseBodyStart(call: Call) {
//                        val headerStart = System.currentTimeMillis()
//                        if(call.request().url.toString().contains("get-model")) {
//                            measurementCallback("download_time_start", "$headerStart")
//                        }

//                        val bytesEnd = System.currentTimeMillis()
//
//                        if(call.request().url.toString().contains("report")) {
//                            measurementCallback("report_time_end", "$bytesEnd")
//                        } else if(call.request().url.toString().contains("get-model")) {
//                            measurementCallback("download_time_end", "$bytesEnd")
//                            measurementCallback("model_download_size", "${(byteCount / 1024) / 1024}")
//                        }
//                    }

//                    override fun requestHeadersStart(call: Call) {
//
//                    }

//                    override fun responseHeadersStart(call: Call) {
//                        val headerStart = System.currentTimeMillis()
//                        if(call.request().url.toString().contains("report")) {
////                            measurementCallback("report_time_start", "$headerStart")
//                        } else if(call.request().url.toString().contains("get-model")) {
//                            Log.e("HttpClient", "$headerStart")
////                            measurementCallback("download_time_start", "$headerStart")
//                        }
//
//                    }

                    override fun requestBodyEnd(call: Call, byteCount: Long) {
                        if(call.request().url.toString().contains("report")) {
//                            Log.e("HttpClient", "bytes sent $byteCount")
                            measurementCallback("model_upload_size", "${(byteCount / 1024) / 1024}")
                        }

                    }

//                    override fun requestBodyStart(call: Call) {
//                        val headerStart = System.currentTimeMillis()
//                        if(call.request().url.toString().contains("report")) {
//                            measurementCallback("report_time_start", "$headerStart")
//                        }
//                    }
                })
                .addNetworkInterceptor {chain ->
                    val request = chain.request().newBuilder()
                        .build()

                    val t1 = System.currentTimeMillis()
                    if(request.url.toString().contains("get-model")) {
                        measurementCallback("download_time_start", "$t1")
                    } else if(request.url.toString().contains("report")) {
                        measurementCallback("report_time_start", "$t1")
                    }

                    val response = chain.proceed(request)

                    response
                }

            val okHttpClient = okHttpClientBuilder.build()
            val apiClient: HttpAPI = Retrofit.Builder()
                .addCallAdapterFactory(RxJava2CallAdapterFactory.create())
                .addConverterFactory(Json.asConverterFactory("application/json".toMediaType()))
                .baseUrl("${NetworkingProtocol.HTTP}://$baseUrl")
                .client(okHttpClient)

                .build().create(HttpAPI::class.java)


            val httpClient = HttpClient(apiClient)
            return httpClient
        }

        fun setMeasurementCallback(callback: (String, String) -> Unit) {
            this.measurementCallback = callback
        }
    }


}