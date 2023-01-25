package org.openmined.syft.demo.fcm

import android.util.Log
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage
import org.greenrobot.eventbus.EventBus
import org.json.JSONException
import org.json.JSONObject
import org.openmined.syft.demo.BuildConfig
import org.openmined.syft.demo.federated.service.FederatedWorker
import org.openmined.syft.demo.federated.ui.main.AUTH_TOKEN
import org.openmined.syft.demo.federated.ui.main.BASE_URL
import org.openmined.syft.demo.utils.AppPreferences
import org.openmined.syft.demo.utils.getCycleStartPushKey
import org.openmined.syft.demo.utils.isInTraining
import org.openmined.syft.demo.utils.setCycleStartPushKey
import timber.log.Timber


class MyFirebaseMessagingService: FirebaseMessagingService() {
    /**
     * Called if the FCM registration token is updated. This may occur if the security of
     * the previous token had been compromised. Note that this is called when the
     * FCM registration token is initially generated so this is where you would retrieve the token.
     */
    override fun onNewToken(token: String) {
        Timber.d("Refreshed token: $token")

        // If you want to send messages to this application instance or
        // manage this apps subscriptions on the server side, send the
        // FCM registration token to your app server.
//        sendRegistrationToServer(token)
    }

    override fun onMessageReceived(remoteMessage: RemoteMessage) {
        // [START_EXCLUDE]
        // There are two types of messages data messages and notification messages. Data messages are handled
        // here in onMessageReceived whether the app is in the foreground or background. Data messages are the type
        // traditionally used with GCM. Notification messages are only received here in onMessageReceived when the app
        // is in the foreground. When the app is in the background an automatically generated notification is displayed.
        // When the user taps on the notification they are returned to the app. Messages containing both notification
        // and data payloads are treated as notification messages. The Firebase console always sends notification
        // messages. For more see: https://firebase.google.com/docs/cloud-messaging/concept-options
        // [END_EXCLUDE]

        // TODO(developer): Handle FCM messages here.
        // Not getting messages here? See why this may be: https://goo.gl/39bRNJ
        Timber.d(" ========================== >>> From: ${remoteMessage.from}")
        Log.e("MyFirebaseService", " ========================== >>> message:")


        // Check if message contains a data payload.
        if (!remoteMessage.data.isNullOrEmpty()) {
            val params:   Map<String, String> = remoteMessage.data
            val objData = JSONObject(params)
            Log.e("MyFirebaseService", " ========================== >>> message: $objData")
            var messageType = objData.getString("type")
            when(messageType) {
                "request_for_cycle","cycle_completed" -> {
                    Log.e("MyFirebaseService", " isInTraining() ${isInTraining()}")
                    if(isInTraining()) return
                    AppPreferences.putBoolean(PushEventType.CYCLE_REQUEST.name, false)
                    val cycleStartKey = JSONObject(objData.getString("data")).getString("cycle_start_request_key")
                    if(getCycleStartPushKey() != cycleStartKey) {
                        setCycleStartPushKey(cycleStartKey)
                        EventBus.getDefault().post(PushEvent(PushEventType.CYCLE_REQUEST))
                    }
                }
                "interrupt_cycle_completed" -> {
                    AppPreferences.putBoolean(AppPreferences.interrupt_training, true)
                }
                "fl_completed" -> {
                    EventBus.getDefault().post(PushEvent(PushEventType.PROCESS_COMPLETED))
                    AppPreferences.putBoolean(PushEventType.PROCESS_COMPLETED.name, false)
                }
                "worker_online_status" -> {
                    EventBus.getDefault().post(PushEvent(PushEventType.TYPE_WORKER_STATUS))
                    AppPreferences.putBoolean(PushEventType.TYPE_WORKER_STATUS.name, false)
                }
                else -> {}

            }
        }

        // Check if message contains a notification payload.
        remoteMessage.notification?.let {
//            Log.d(TAG, "Message Notification Body: ${it.body}")
        }

        // Also if you intend on generating your own notifications as a result of a received FCM
        // message, here is where that should be initiated. See sendNotification method below.


    }

    fun sendRegistrationTokenToServer(token: String) {

    }

    override fun onMessageSent(p0: String) {
        super.onMessageSent(p0)
    }
}