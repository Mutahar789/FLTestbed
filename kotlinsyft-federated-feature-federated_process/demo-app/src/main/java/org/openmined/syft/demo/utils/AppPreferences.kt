package org.openmined.syft.demo.utils

import android.content.Context
import android.content.SharedPreferences
import org.openmined.syft.demo.MyApp

class AppPreferences {
    companion object {
        var interrupt_training = "interrupt_training"
        fun getPrefEditor(): SharedPreferences {
            return MyApp.getInstance().getSharedPreferences("FL_Evaluation", Context.MODE_PRIVATE)
        }

        fun putString(key: String, value: String) {
            with (getPrefEditor().edit()) {
                putString(key, value)
                apply()
            }
        }

        fun getString(key: String): String {
            return getPrefEditor().getString(key, "")!!
        }

        fun putBoolean(key: String, value: Boolean) {
            with (getPrefEditor().edit()) {
                putBoolean(key, value)
                apply()
            }
        }

        fun getBoolean(key: String): Boolean {
            return getPrefEditor().getBoolean(key, false)!!
        }


    }
}