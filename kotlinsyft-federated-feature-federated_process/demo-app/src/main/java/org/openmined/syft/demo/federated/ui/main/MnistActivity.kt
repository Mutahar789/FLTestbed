package org.openmined.syft.demo.federated.ui.main

import android.Manifest
import android.app.ActivityManager
import android.content.ContentValues
import android.content.Context
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.ProgressBar
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.databinding.DataBindingUtil
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProvider
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.jaredrummler.android.processes.AndroidProcesses
import com.jaredrummler.android.processes.models.AndroidAppProcess
import kotlinx.android.synthetic.main.activity_mnist.*
import org.greenrobot.eventbus.EventBus
import org.greenrobot.eventbus.Subscribe
import org.greenrobot.eventbus.ThreadMode
import org.openmined.syft.Syft
import org.openmined.syft.demo.BuildConfig
import org.openmined.syft.demo.MyApp
import org.openmined.syft.demo.R
import org.openmined.syft.demo.databinding.ActivityMnistBinding
import org.openmined.syft.demo.fcm.PushEvent
import org.openmined.syft.demo.fcm.PushEventType
import org.openmined.syft.demo.federated.datasource.LocalFEMNISTDataSource
import org.openmined.syft.demo.federated.datasource.LocalMNISTDataDataSource
import org.openmined.syft.demo.federated.datasource.LocalSyntheticDataSource
import org.openmined.syft.demo.federated.domain.FEMNISTDataRepository
import org.openmined.syft.demo.federated.domain.MNISTDataRepository
import org.openmined.syft.demo.federated.domain.SyntheticDataRepository
import org.openmined.syft.demo.federated.service.TrainService
import org.openmined.syft.demo.federated.service.WorkerRepository
import org.openmined.syft.demo.federated.ui.ContentState
import org.openmined.syft.demo.federated.ui.ProcessData
import org.openmined.syft.demo.login.LoginViewModel
import org.openmined.syft.demo.login.LoginViewModelFactory
import org.openmined.syft.demo.utils.*
import org.openmined.syft.domain.SyftConfiguration
import timber.log.Timber
import java.io.File
import java.io.IOException
import java.text.DateFormat
import java.text.SimpleDateFormat
import java.util.*
import kotlin.collections.ArrayList
import kotlin.concurrent.timerTask


const val AUTH_TOKEN = "authToken"
const val SYFT_WORKER_ID = "syft_worker_id"
const val BASE_URL = "baseUrl"
private const val TAG = "MnistActivity"

enum class DATASET {
    MNIST, SYNTHETIC, FEMNIST
}

var pids = intArrayOf(0, 0)

class MnistActivity : AppCompatActivity() {
    var recordStats = true

    private lateinit var binding: ActivityMnistBinding
    private lateinit var viewModel: MnistActivityViewModel
    private lateinit var loginViewModel: LoginViewModel
    private lateinit var syftConfig: SyftConfiguration

    private lateinit var mnistDataRepository: MNISTDataRepository
    private lateinit var syntheticDataRepository: SyntheticDataRepository
    private lateinit var femnistDataRepository: FEMNISTDataRepository


    private var dataSet = DATASET.FEMNIST

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = DataBindingUtil.setContentView(this, R.layout.activity_mnist)
        binding.lifecycleOwner = this
        setSupportActionBar(toolbar)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        when(dataSet) {
            DATASET.SYNTHETIC -> {
                val localSyntheticDataSource = LocalSyntheticDataSource(resources)
                syntheticDataRepository = SyntheticDataRepository(localSyntheticDataSource)
            }
            DATASET.FEMNIST -> {
                val localFEMNISTDataSource = LocalFEMNISTDataSource(resources)
                femnistDataRepository = FEMNISTDataRepository(localFEMNISTDataSource)
            }

            else -> {
                val mnistDataSource = LocalMNISTDataDataSource(resources)
                mnistDataRepository = MNISTDataRepository(mnistDataSource)
            }
        }

//        val isServiceRunning = isMyServiceRunning()
//        Timber.d("------------------isMyServiceRunning() $isServiceRunning")
//        if(isServiceRunning) {
//            stopService(Intent(this, TrainService::class.java))
//        }
//        val intent = Intent(this, TrainService::class.java)
//        startService(intent)

        val baseUrl = "192.168.100.30:5000"

        syftConfig = SyftConfiguration.builder(this, baseUrl)
            .setMeasurementCallback({ key, value ->
                AppPreferences.putString(key, value)
                Timber.e("$key -> $value")
            })
            .setMessagingClient(SyftConfiguration.NetworkingClients.HTTP)
            .setCacheTimeout(0L)
            .enableBatteryCheck()
            .enableMeteredData()

            .build()

        val arr = ArrayList<String>()
        arr.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        arr.add(Manifest.permission.READ_EXTERNAL_STORAGE)

        ActivityCompat.requestPermissions(
            this,
            arr.toTypedArray(),
            12
        );


        val dir: File = File(getFilesDir(), "mydir")
        if (!dir.exists()) {
            dir.mkdir()
        }
        val gpxfile = File(dir, "memoryinfo.csv")
        if(gpxfile.exists())
            gpxfile.delete()

       android.os.Handler().postDelayed({
           buildMemoryProfile()
       }, 1000)

        loginViewModel = initiateViewModel(syftConfig, baseUrl)
        this.viewModel = initiateViewModel(
            baseUrl,
            BuildConfig.SYFT_AUTH_TOKEN
        )
        binding.viewModel = this.viewModel

        viewModel.getRunningWorkInfo()?.observe(this, viewModel.getWorkInfoObserver())

        viewModel.processState.observe(
            this,
            Observer { onProcessStateChanged(it) }
        )

        viewModel.processData.observe(
            this,
            Observer { onProcessData(it) }
        )

        viewModel.steps.observe(
            this,
            Observer { binding.step.text = it })

        MyApp.getInstance().getDeviceToken { deviceToken ->

            var cacheWorkerId = AppPreferences.getString(SYFT_WORKER_ID)
            cacheWorkerId = ""
            if(cacheWorkerId.isNullOrEmpty()) {
                loginViewModel.authenticate({
                    val workerId = Syft.getInstance(syftConfig, getAuthToken()).getSyftWorkerId()
                    workerId?.let {
                        AppPreferences.putString(SYFT_WORKER_ID, workerId!!)
                    }

                }, getMemorySizeInBytes().toFloat(), getNumberOfCores())

            } else {
                Syft.getInstance(syftConfig, getAuthToken()).setSyftWorkerId(cacheWorkerId)
            }

            try {
                val workerId = Syft.getInstance(syftConfig, getAuthToken()).getSyftWorkerId()
                if (!workerId.isNullOrEmpty()) {
                    sendWorkerStatusToServer()
                }
            } catch (ex: Exception) {
                ex.printStackTrace()
            }

        }
    }

    private fun isMyServiceRunning(): Boolean {
        val manager = getSystemService(ACTIVITY_SERVICE) as ActivityManager
        for (service in manager.getRunningServices(Int.MAX_VALUE)) {
            Timber.d("TrainService::class.simpleName ${TrainService::class.qualifiedName} service.service.className ${service.service.className}")
            if (TrainService::class.qualifiedName == service.service.className) {
                return true
            }
        }
        return false
    }

    fun updateWorkerStatus() {
        MyApp.getInstance().getDeviceToken { deviceToken ->

            var cacheWorkerId = AppPreferences.getString(SYFT_WORKER_ID)
            if(cacheWorkerId.isNullOrEmpty()) {
                loginViewModel.authenticate({
                    val workerId = Syft.getInstance(syftConfig, getAuthToken()).getSyftWorkerId()
                    workerId?.let {
                        AppPreferences.putString(SYFT_WORKER_ID, workerId!!)
                    }

                }, getMemorySizeInBytes().toFloat(), getNumberOfCores())

            } else {
                Syft.getInstance(syftConfig, getAuthToken()).setSyftWorkerId(cacheWorkerId)
            }

            try {
                val workerId = Syft.getInstance(syftConfig, getAuthToken()).getSyftWorkerId()
                if (!workerId.isNullOrEmpty()) {
                    sendWorkerStatusToServer()
                }
            } catch (ex: Exception) {
                ex.printStackTrace()
            }

        }
    }

    private fun launchBackgroundCycle() {
        viewModel.submitJob().observe(this, viewModel.getWorkInfoObserver())
    }

    private fun launchForegroundCycle(config: SyftConfiguration) {
//        val config = SyftConfiguration.builder(this, viewModel.baseUrl)
//                .setMessagingClient(SyftConfiguration.NetworkingClients.HTTP)
//                .setCacheTimeout(0L)
//                .enableBatteryCheck()
//                .enableMeteredData()
//                .build()

        when(dataSet) {
            DATASET.SYNTHETIC -> viewModel.launchForegroundTrainer(
                this,
                config,
                syntheticDataRepository,
                dataSet
            )
            DATASET.FEMNIST -> viewModel.launchForegroundTrainer(
                this,
                config,
                femnistDataRepository,
                dataSet
            )
            else -> viewModel.launchForegroundTrainer(this, config, mnistDataRepository, dataSet)
        }

    }

    override fun onBackPressed() {
        super.onBackPressed()
        viewModel.disposeTraining()
        finish()
    }

    private fun onProcessData(it: ProcessData?) {
        processData(
            it ?: ProcessData(
                emptyList()
            )
        )
    }

    private fun onProcessStateChanged(contentState: ContentState?) {
        when (contentState) {
            ContentState.Training, ContentState.REJECTION -> {
                progressBar.visibility = ProgressBar.GONE
//                binding.chartHolder.visibility = View.VISIBLE
                binding.llCycleDetails.visibility = View.VISIBLE
            }
            ContentState.Loading -> {
                progressBar.visibility = ProgressBar.VISIBLE
                binding.chartHolder.visibility = View.GONE
                binding.llCycleDetails.visibility = View.GONE
            }
        }
    }

    private fun processData(processState: ProcessData) {
        val entries = mutableListOf<Entry>()
        processState.data.forEachIndexed { index, value ->
            entries.add(Entry(index.toFloat(), value))
        }
        val dataSet = LineDataSet(entries, "test accuracy")
        val lineData = LineData(dataSet)
        chart.data = lineData
        chart.setMaxVisibleValueCount(0)
        chart.setNoDataText("Waiting for data")
        chart.invalidate()
    }

    private fun initiateViewModel(baseUrl: String?, authToken: String?): MnistActivityViewModel {
        if (baseUrl == null || authToken == null)
            throw IllegalArgumentException("Mnist trainer called without proper arguments")
        return ViewModelProvider(
            this,
            MnistViewModelFactory(
                baseUrl,
                authToken,
                WorkerRepository(this)
            )
        ).get(MnistActivityViewModel::class.java)
    }

    @Subscribe(threadMode = ThreadMode.MAIN)
    fun onMessageEvent(pushEvent: PushEvent) {
        when(pushEvent.eventType) {
            PushEventType.CYCLE_REQUEST, PushEventType.CYCLE_COMPLETED -> {
                if (isInTraining()) {
                    return
                }

                viewModel.disposeTraining()

                var cacheWorkerId = AppPreferences.getString(SYFT_WORKER_ID)
                Syft.getInstance(syftConfig, getAuthToken()).setSyftWorkerId(cacheWorkerId)
                launchForegroundCycle(syftConfig)
//                launchBackgroundCycle()
            }
            PushEventType.PROCESS_COMPLETED -> {
                recordStats = false
                binding.chartHolder.visibility = View.GONE
                binding.llCycleDetails.visibility = View.GONE
                binding.scrollArea.visibility = View.GONE
                binding.tvFlCompleted.visibility = View.VISIBLE
//                showMessage("Training has been completed.")
            }
            PushEventType.TYPE_WORKER_STATUS -> {
//                sendWorkerStatusToServer()
                updateWorkerStatus()
            }
        }

        AppPreferences.putBoolean(pushEvent.eventType.name, true)
    }

    fun sendWorkerStatusToServer() {

        viewModel.sendWorkerStatusToServer(syftConfig)
    }

    override fun onStart() {
        super.onStart()
        EventBus.getDefault().register(this)
    }

    override fun onStop() {
        EventBus.getDefault().unregister(this)
        super.onStop()
    }

    fun checkForMissingPushNotificaiton() {
        if(AppPreferences.getBoolean(PushEventType.TYPE_WORKER_STATUS.name)) {
            onMessageEvent(PushEvent(PushEventType.TYPE_WORKER_STATUS))
        } else if(AppPreferences.getBoolean(PushEventType.CYCLE_REQUEST.name) ) {
            onMessageEvent(PushEvent(PushEventType.CYCLE_REQUEST))
        } else if(AppPreferences.getBoolean(PushEventType.CYCLE_COMPLETED.name) ) {
//            onMessageEvent(PushEvent(PushEventType.CYCLE_REQUEST))
        }
    }

    private fun initiateViewModel(config: SyftConfiguration, baseUrl: String): LoginViewModel {
        return ViewModelProvider(this, LoginViewModelFactory(config, MyApp.getInstance(), baseUrl)).get(
            LoginViewModel::class.java
        )
    }

    private fun buildMemoryProfile() {
        pids[0] = android.os.Process.myPid()
        try {
            val p1 = Runtime.getRuntime()
                    .exec(arrayOf("su", "-c", "toybox renice -n -30 -p " + pids.get(0)))
            p1.waitFor()
            p1.destroy()
        } catch (e2: IOException) {
            e2.printStackTrace()
        } catch (e22: InterruptedException) {
            e22.printStackTrace()
        }
        val processName = "org.openmined.syft.demo"
//        val processName = "com.example.myapplication"
        val am = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
//        Timber.d("\n======================================================\n")
//        Timber.d("am.memoryClass.toString() ${am.memoryClass}")

        val memoryInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memoryInfo)
//        Timber.d("memoryInfo.threshold ${memoryInfo.threshold} memoryInfo.availMem ${memoryInfo.availMem}")
//
//        Timber.d("am.memoryClass.toString() ${am.memoryClass}")
//        Timber.d("\n======================================================\n")
//        Timber.d("\n======================================================\n")
        val it: Iterator<AndroidAppProcess> = AndroidProcesses.getRunningAppProcesses().iterator()
        while (true) {
            if (!it.hasNext()) {
                break
            }
            val process = it.next()
            if (process.name == processName) {
                pids[1] = process.stat().pid
                break
            }
        }

        val processes = AndroidProcesses.getRunningAppProcesses()
        for (process in processes) {
            if (pids[0] == process.stat().pid && process.oom_adj() >= 0) {
                Timber.e("Changing OOM " + pids[0])
                val cmd = arrayOf("su", "-c", "echo -17 > /proc/" + pids[0] + "/oom_adj")
                val p = Runtime.getRuntime().exec(cmd)
                p.waitFor()
                p.destroy()
                break
            }
        }


        SaveText(
            "memoryinfo.csv",
            "time,pressure_pss(MB)," + processName + "_pss(MB),Active_Memory(MB),Cached_Memory(MB),Free_Memory(MB),SwapCached_Memory(MB),Inactive:,Active(anon):,Inactive(anon):,Active(file):, Inactive(file):,Unevictable:,Mlocked:,SwapTotal:,SwapFree:,Dirty:,Writeback:,AnonPages:,Mapped:,Shmem:,Slab:,SReclaimable:,SUnreclaim:,KernelStack:,PageTables:,NFS_Unstable:,Bounce:,WritebackTmp:,CommitLimit:,Committed_AS:,VmallocTotal:,VmallocUsed:,VmallocChunk:,totalPrivateClean,totalPrivateDirty,totalSharedClean,totalSharedDirty,totalSwappablePss\n",
            this
        );

        var counter = 0

        Timer().scheduleAtFixedRate(timerTask {
            if (recordStats) {
                val list = findMemoryStats(
                    getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager,
                    pids
                )

                val initial_Cache = list!![3] + list[4]
                val vmPressure =
                    ((initial_Cache - (list[3] + list[4])) * 100 / initial_Cache).toDouble()
                val dateFormat: DateFormat = SimpleDateFormat("HH:mm:ss")

                SaveText(
                    "memoryinfo.csv",
                    dateFormat.format(Date())
                        .toString() + "," + list!![0] + "," + list[1] + "," + list[2] + "," + list[3] + "," + list[4] + "," + list[6] + "," +
                            list[7] + "," + list[8] + "," + list[9] + "," + list[10] + "," + list[11] + "," + list[12] + "," + list[13] + "," + list[14] + "," +
                            list[15] + "," + list[16] + "," + list[17] + "," + list[18] + "," + list[19] + "," + list[20] + "," + list[21] + "," + list[22] + "," + list[23] + "," +
                            list[24] + "," + list[25] + "," + list[26] + "," + list[27] + "," + list[28] + "," + list[29] + "," + list[30] + "," + list[31] + "," + list[32] + "," + list[33]
                            + "," + list[34]
                            + "," + list[35]
                            + "," + list[36]
                            + "," + list[37]
                            + "," + list[38] + "," + "\n",
                    this@MnistActivity
                )
            } else {
                this.cancel()
            }

        }, 1000, 1000)



    }
}


