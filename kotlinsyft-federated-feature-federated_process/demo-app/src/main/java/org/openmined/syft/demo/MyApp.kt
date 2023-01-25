package org.openmined.syft.demo

import android.app.Application
import com.google.android.gms.tasks.OnCompleteListener
import com.google.firebase.messaging.FirebaseMessaging
import org.openmined.syft.demo.utils.AppPreferences
import org.openmined.syft.demo.utils.setCycleStartPushKey
import org.openmined.syft.demo.utils.setTrainingMode
import timber.log.Timber
import java.util.Random

class MyApp : Application() {
    override fun onCreate() {
        super.onCreate()
        instance = this
        Timber.plant(Timber.DebugTree())

        var currenToken = FirebaseMessaging.getInstance().getToken()

        setTrainingMode(false)
        setCycleStartPushKey("")
        AppPreferences.putString("report_time", "")
        AppPreferences.putBoolean("interrupt_training", false)
        AppPreferences.putString("report_time_end", "")
        AppPreferences.putString("report_time_start", "")
        AppPreferences.putString("download_time_end", "")
        AppPreferences.putString("download_time_start", "")
        Timber.d(" ===== Current Token =>>>> $currenToken")
    }

    companion object {
        private var instance: MyApp ? = null

        fun getInstance(): MyApp {
            if(instance == null) {
                synchronized(MyApp::class.java) {
                    if(instance == null) {
                        instance = MyApp()
                    }
                }
            }

            return instance!!
        }


    }

    fun getDeviceToken(callback: (String) -> Unit) {
        FirebaseMessaging.getInstance().token.addOnCompleteListener(OnCompleteListener { task ->
            if (!task.isSuccessful) {
                Timber.d("Fetching FCM registration token failed" + task.exception)
                return@OnCompleteListener
            }

            // Get new FCM registration token
            callback(task.result!!)

//            Timber.d( msg)
            Timber.d(" ===== App.Kt New Token =>>>> ${task.result!!}")
        })
    }
}