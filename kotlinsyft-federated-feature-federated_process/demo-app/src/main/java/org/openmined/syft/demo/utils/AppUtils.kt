package org.openmined.syft.demo.utils

import android.app.Activity
import android.app.ActivityManager
import android.app.ActivityManager.RunningAppProcessInfo
import android.content.Context
import android.content.Context.ACTIVITY_SERVICE
import android.os.Debug
import android.util.Log
import android.widget.Toast
import org.openmined.syft.demo.BuildConfig
import org.openmined.syft.demo.MyApp
import org.pytorch.IValue
import timber.log.Timber
import java.io.*
import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.collections.set

const val MODEL_NAME = "mnist"
const val MODEL_VERSION = "1.0"
const val IN_TRAINING = "in_training_mode"
const val CYCLE_START_PUSH_KEY = "cycle_start_push_key"

fun showMessage(message: String) {
    Toast.makeText(MyApp.getInstance(), message, Toast.LENGTH_SHORT).show()
}

fun getAuthToken() : String = BuildConfig.SYFT_AUTH_TOKEN

fun setTrainingMode(trainingMode: Boolean) {
    AppPreferences.putBoolean(IN_TRAINING, trainingMode)
}

fun isInTraining() = AppPreferences.getBoolean(IN_TRAINING)

fun getCycleStartPushKey() = AppPreferences.getString(CYCLE_START_PUSH_KEY)

fun setCycleStartPushKey(value: String) = AppPreferences.putString(CYCLE_START_PUSH_KEY, value)

fun getMemoryInfo(activity: Activity) {
    val mi = ActivityManager.MemoryInfo()
    val activityManager = activity.getSystemService(ACTIVITY_SERVICE) as ActivityManager?
    activityManager!!.getMemoryInfo(mi)


//    var text = if (epStartTime > 0) {
//        "$epStartTime,$epEndTime,${epEndTime - epStartTime},${mi.totalMem / 0x100000L},${(mi.availMem / 0x100000L).toDouble()},$loss,$accuracy\n"
//    } else {
//        "$loss,$accuracy\n"
//    }
}

fun getMemorySizeInBytes(): Float {
    val activityManager = MyApp.getInstance().getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val memoryInfo = ActivityManager.MemoryInfo()
    activityManager.getMemoryInfo(memoryInfo)
    val totalMemory = memoryInfo.totalMem

    val kb = totalMemory / 1024.0
    val mb = totalMemory / 1048576.0
    val gb = totalMemory / 1073741824.0

    Timber.d("===================== totalRAM is ${totalMemory} kb $kb mb $mb gb $gb")

    return gb.toFloat()
}

fun getNumberOfCores(): Int {
    return Runtime.getRuntime().availableProcessors()
}

fun shuffle(arr: ArrayList<ArrayList<Float>>, n: Int, seed: Long) {
    // Creating a object for Random class
    val r = Random()
    r.setSeed(seed)
    // Start from the last element and swap one by one. We don't
    // need to run for the first element that's why i > 0
    for (i in n-1 downTo 1) {

        // Pick a random index from 0 to i
        val j: Int = r.nextInt(i + 1)

        // Swap arr[i] with the element at random index
        val temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp
    }
}

fun shuffleLabels(arr: ArrayList<Long>, n: Int, seed: Long) {
    // Creating a object for Random class
    val r = Random()
    r.setSeed(seed)
    // Start from the last element and swap one by one. We don't
    // need to run for the first element that's why i > 0
    for (i in n-1 downTo 1) {

        // Pick a random index from 0 to i
        val j: Int = r.nextInt(i + 1)

        // Swap arr[i] with the element at random index
        val temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp
    }
}

fun printIValue(iValue: IValue, shouldCount:Boolean = true) {
    println()
    var count = 0
    for(item in iValue.toTensor().dataAsFloatArray) {
        print("$ $item ")
        count += 1
        if(count % 5 == 0 && shouldCount)
            print("\n")
    }
    println()
}

fun findMemoryStats(activityManager: ActivityManager, pids: IntArray): List<Int>? {
    val memoryInfoArray: Array<Debug.MemoryInfo> = activityManager.getProcessMemoryInfo(pids)
    return try {
        val runtime = Runtime.getRuntime()
        val memMap = getStringFromInputStream(
            runtime.exec("cat /proc/meminfo " + pids.get(1)).inputStream,
            2
        )

        ActivityManager.getMyMemoryState(RunningAppProcessInfo())
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        if (memoryInfo.lowMemory) {
            Log.e("ContentValues", "low memory and threshold:" + memoryInfo.threshold)
        }
        val list: MutableList<Int> = ArrayList()

//        Timber.d("============================= getTotalSwappablePss() ${memoryInfoArray[0].totalSwappablePss}")
        list.add(Integer.valueOf(memoryInfoArray[0].getTotalPss() / 1024))
        list.add(Integer.valueOf(memoryInfoArray[1].getTotalPss() / 1024))
        list.add(Integer.valueOf(memMap["Active:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Cached:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["MemFree:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["MemTotal:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["SwapCached:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Inactive:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Active(anon):"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Inactive(anon):"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Active(file):"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Inactive(file):"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Unevictable:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Mlocked:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["SwapTotal:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["SwapFree:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Dirty:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Writeback:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["AnonPages:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Mapped:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Shmem:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Slab:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["SReclaimable:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["SUnreclaim:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["KernelStack:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["PageTables:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["NFS_Unstable:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Bounce:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["WritebackTmp:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["CommitLimit:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["Committed_AS:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["VmallocTotal:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["VmallocUsed:"]!!.toInt() / 1024))
        list.add(Integer.valueOf(memMap["VmallocChunk:"]!!.toInt() / 1024))


        list.add(Integer.valueOf(memoryInfoArray[0].totalPrivateClean / 1024))
        list.add(Integer.valueOf(memoryInfoArray[0].totalPrivateDirty / 1024))
        list.add(Integer.valueOf(memoryInfoArray[0].totalSharedClean / 1024))
        list.add(Integer.valueOf(memoryInfoArray[0].totalSharedDirty / 1024))
        list.add(Integer.valueOf(memoryInfoArray[0].totalSwappablePss / 1024))

        list
    } catch (e: IOException) {
        e.printStackTrace()
        emptyList<Int>()
    }
}

fun SaveText(sFileName: String?, sBody: String?, activity: Activity) {
//    try {
//        val file = File("/sdcard/$sFileName}")
//
//        val writer = FileWriter(file, true)
//        writer.append(sBody)
//        writer.flush()
//        writer.close()
//    } catch (e: IOException) {
//        e.printStackTrace()
//        Log.d("ContentValues", e.printStackTrace().toString())
//    }

    val dir: File = File(activity.getFilesDir(), "mydir")
    if (!dir.exists()) {
        dir.mkdir()
    }

    try {
        val gpxfile = File(dir, sFileName)
        val writer = FileWriter(gpxfile, true)
        writer.append(sBody)
        writer.flush()
        writer.close()
    } catch (e: Exception) {
        e.printStackTrace()
    }
}

private fun getStringFromInputStream(inputStream: InputStream, oneLine: Int): Map<String, Int> {
    var str: String
    var sb: StringBuilder
    val sb2 = StringBuilder()
    val br = BufferedReader(InputStreamReader(inputStream))
    val Map: MutableMap<String, Int> = HashMap()
    while (true) {
        try {
            val readLine: String? = br.readLine()
            if (readLine != null) {
                val strs = readLine.split(" ".toRegex()).toTypedArray()
                if (oneLine == 0) {
                    Map["pid"] = Integer.valueOf(strs[1].toInt())
                } else if (oneLine == 1) {
                    Map["utime"] = Integer.valueOf(strs[13].toInt())
                    Map["stime"] = Integer.valueOf(strs[14].toInt())
                    Map["cutime"] = Integer.valueOf(strs[15].toInt())
                    Map["cstime"] = Integer.valueOf(strs[16].toInt())
                    Map["starttime"] = Integer.valueOf(strs[21].toInt())
                } else if (oneLine == 2) {
                    Map[strs[0]] = Integer.valueOf(strs[strs.size - 2].toInt())
                }
                sb2.append(readLine)
                sb2.append("\n")
            } else {
                try {
                    break
                } catch (e: IOException) {
//                    e = e
                    str = "ContentValues"
                    sb = StringBuilder()
                }
            }
        } catch (e2: IOException) {
            Log.e("ContentValues", "------ getStringFromInputStream " + e2.message)
            try {
                br.close()
            } catch (e3: IOException) {
//                e = e3
                str = "ContentValues"
                sb = StringBuilder()
            }
        } catch (th: Throwable) {
            try {
                br.close()
            } catch (e4: IOException) {
                Log.e("ContentValues", "------ getStringFromInputStream " + e4.message)
            }
            throw th
        }
    }
    br.close()
    return Map
    sb.append("------ getStringFromInputStream ")
//    sb.append(e.getMessage())
    Log.e(str, sb.toString())
    return Map
}