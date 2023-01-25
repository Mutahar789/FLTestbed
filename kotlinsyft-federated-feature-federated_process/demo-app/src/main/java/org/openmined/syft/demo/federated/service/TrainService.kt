package org.openmined.syft.demo.federated.service

import android.Manifest
import android.app.*
import android.app.Service.START_STICKY
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.util.Log
import android.view.View
import androidx.annotation.RequiresApi
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModelProvider
import androidx.work.ForegroundInfo
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.workDataOf
import io.reactivex.disposables.CompositeDisposable
import org.greenrobot.eventbus.EventBus
import org.greenrobot.eventbus.Subscribe
import org.greenrobot.eventbus.ThreadMode
import org.openmined.syft.Syft
import org.openmined.syft.demo.BuildConfig
import org.openmined.syft.demo.MyApp
import org.openmined.syft.demo.fcm.PushEvent
import org.openmined.syft.demo.fcm.PushEventType
import org.openmined.syft.demo.federated.datasource.LocalFEMNISTDataSource
import org.openmined.syft.demo.federated.datasource.LocalMNISTDataDataSource
import org.openmined.syft.demo.federated.datasource.LocalSyntheticDataSource
import org.openmined.syft.demo.federated.domain.FEMNISTDataRepository
import org.openmined.syft.demo.federated.domain.MNISTDataRepository
import org.openmined.syft.demo.federated.domain.SyntheticDataRepository
import org.openmined.syft.demo.federated.domain.TrainingTask
import org.openmined.syft.demo.federated.logging.MnistLogger
import org.openmined.syft.demo.federated.ui.ContentState
import org.openmined.syft.demo.federated.ui.ProcessData
import org.openmined.syft.demo.federated.ui.main.DATASET
import org.openmined.syft.demo.federated.ui.main.MnistActivityViewModel
import org.openmined.syft.demo.federated.ui.main.MnistViewModelFactory
import org.openmined.syft.demo.federated.ui.main.SYFT_WORKER_ID
import org.openmined.syft.demo.federated.ui.work.WorkInfoActivity
import org.openmined.syft.demo.login.LoginViewModel
import org.openmined.syft.demo.utils.*
import org.openmined.syft.domain.SyftConfiguration
import org.openmined.syft.execution.JobStatusSubscriber
import org.openmined.syft.execution.Plan
import org.openmined.syft.networking.datamodels.ClientConfig
import org.openmined.syft.proto.SyftModel
import timber.log.Timber
import java.io.File
import java.util.concurrent.ConcurrentHashMap

class TrainService : Service() {

    private val mHandler: Handler? = Handler()

    private lateinit var mRunnable: Runnable
    private lateinit var syftConfig: SyftConfiguration

    private lateinit var mnistDataRepository: MNISTDataRepository
    private lateinit var syntheticDataRepository: SyntheticDataRepository
    private lateinit var femnistDataRepository: FEMNISTDataRepository


    private var dataSet = DATASET.FEMNIST
    private var syftWorker: Syft? = null

    private val compositeDisposable = CompositeDisposable()
    private var trainingTask: TrainingTask? = null

    override fun onCreate() {
        super.onCreate()
        startForegroundService()
        EventBus.getDefault().register(this)
    }

    override fun onBind(p0: Intent?): IBinder? {
        return null;
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {

        Log.d(TAG, "ON START COMMAND")

//        if (intent != null) {
//
//            when (intent.action) {
//
//                ACTION_STOP_FOREGROUND_SERVICE -> {
//                    stopService()
//                }
//
//                ACTION_OPEN_APP -> openAppHomePage(intent.getStringExtra(KEY_DATA))
//            }
//        }
        return START_STICKY;
    }


    private fun openAppHomePage(value: String) {

//        val intent = Intent(this, MainActivity::class.java)
//        intent.putExtra(KEY_DATA, value)
//        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
//        startActivity(intent)

    }




    /* Used to build and start foreground service. */
    private fun startForegroundService() {

        //Create Notification channel for all the notifications sent from this app.
        val foregroundInfo = createForegroundInfo("0")

        // Start foreground service.
        startFService(foregroundInfo)


        mRunnable = Runnable {
            Timber.d("------------ App is running")
//            mHandler?.postDelayed(mRunnable, ONE_MIN_MILLI)  //repeat
        };

        // Schedule the task to repeat after 1 second
        mHandler?.postDelayed(
            mRunnable, // Runnable
            ONE_MIN_MILLI // Delay in milliseconds
        )

        Timber.d("------------ startForegroundService end")

        startTrainingProcess()
    }

    private fun startTrainingProcess() {
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

        val baseUrl = "192.168.100.30:5000"

        syftConfig = SyftConfiguration.builder(this, baseUrl)
            .setMessagingClient(SyftConfiguration.NetworkingClients.HTTP)
            .setCacheTimeout(0L)
            .enableBatteryCheck()
            .enableMeteredData()
            .build()

//        val arr = ArrayList<String>()
//        arr.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
//        arr.add(Manifest.permission.READ_EXTERNAL_STORAGE)
//
//        ActivityCompat.requestPermissions(
//            this,
//            arr.toTypedArray(),
//            12
//        );


        val dir: File = File(getFilesDir(), "mydir")
        if (!dir.exists()) {
            dir.mkdir()
        }
        val gpxfile = File(dir, "memoryinfo.csv")
        if(gpxfile.exists())
            gpxfile.delete()

//        android.os.Handler().postDelayed({
//            buildMemoryProfile()
//        }, 1000)
//
//
//
//        loginViewModel = initiateViewModel(syftConfig, baseUrl)
//        this.viewModel = initiateViewModel(
//            baseUrl,
//            BuildConfig.SYFT_AUTH_TOKEN
//        )


        MyApp.getInstance().getDeviceToken { deviceToken ->
            Timber.d("----------------- Device token found")
            var cacheWorkerId = AppPreferences.getString(SYFT_WORKER_ID)
            cacheWorkerId = ""
            if(cacheWorkerId.isNullOrEmpty()) {
                authenticate({
                    val workerId = Syft.getInstance(syftConfig, getAuthToken()).getSyftWorkerId()
                    workerId?.let {
                        AppPreferences.putString(SYFT_WORKER_ID, workerId!!)
                    }

                }, getMemorySizeInBytes().toFloat(), getNumberOfCores())

            } else {
                Syft.getInstance(syftConfig, getAuthToken()).setSyftWorkerId(cacheWorkerId)
            }

//            try {
//                val workerId = Syft.getInstance(syftConfig, getAuthToken()).getSyftWorkerId()
//                if (!workerId.isNullOrEmpty()) {
//                    sendWorkerStatusToServer()
//                }
//            } catch (ex: Exception) {
//                ex.printStackTrace()
//            }

        }
    }

    private fun authenticate(callback: (Throwable?) -> Unit, ramSize: Float, cpuCores: Int) {
        Timber.d("----------------- authenticate")
        MyApp.getInstance().getDeviceToken {deviceToken ->
            syftWorker = Syft.getInstance(syftConfig, getAuthToken())
            val token = deviceToken
            Timber.d("================== token $token")
            val mnistJob = syftWorker!!.newJob(MODEL_NAME, MODEL_VERSION, token)

            Timber.d("Aunthentication job started \n")

            val jobStatusSubscriber = object : JobStatusSubscriber() {
                override fun onReady(
                    model: SyftModel,
                    CycleId: String,
                    plans: ConcurrentHashMap<String, Plan>,
                    clientConfig: ClientConfig
                ) {
                    Timber.d("========================================= onready")
                    Timber.d("Token has been saved on server")
//                    syftWorker?.dispose()
                    mnistJob.markJobCompleted()
                    callback(null)

//                    syftWorker?.dispose()
                }

                override fun onComplete() {
                    Timber.d("========================================= oncomplete")
                    mnistJob.markJobCompleted()
                    callback(null)
                }

                override fun onError(throwable: Throwable) {
                    Timber.d("========================================= onerror")
                    throwable.printStackTrace()
                    mnistJob.markJobCompleted()
                    callback(throwable)
                }
            }

            mnistJob.executeFirebaseTokenRequest(jobStatusSubscriber, ramSize, cpuCores)
        }
    }

    private fun createForegroundInfo(progress: String): Notification {
//        val intent = WorkManager.getInstance(applicationContext)
//            .createCancelPendingIntent("training")
        val notifyIntent = Intent(applicationContext, WorkInfoActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }
        val notifyPendingIntent = PendingIntent.getActivity(
            applicationContext, 0, notifyIntent, PendingIntent.FLAG_ONE_SHOT
        )

        Timber.d("--------------Build.VERSION.SDK_INT ${Build.VERSION.SDK_INT}")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            createChannel()
        }


        val notification = NotificationCompat.Builder(applicationContext,
            CHANNEL_ID
        )
            .setContentTitle("Federated Trainer")
//                .setProgress(100, progress, false)
            .setTicker("Federated Trainer")
            .setContentIntent(notifyPendingIntent)
            .setContentText("$progress")
            .setSmallIcon(android.R.drawable.ic_menu_manage)
            .setOngoing(true)
            .build()

        return notification//ForegroundInfo(NOTIFICATION_ID, notification)
    }

    private fun disposeTraining() {
        compositeDisposable.clear()
        trainingTask?.disposeTraining()
    }

    @Subscribe(threadMode = ThreadMode.MAIN)
    fun onMessageEvent(pushEvent: PushEvent) {
        when(pushEvent.eventType) {
            PushEventType.CYCLE_REQUEST, PushEventType.CYCLE_COMPLETED -> {
                if(isInTraining()) {
                    return
                }

                var cacheWorkerId = AppPreferences.getString(SYFT_WORKER_ID)
                Syft.getInstance(syftConfig, getAuthToken()).setSyftWorkerId(cacheWorkerId)
                launchForegroundCycle()
//                launchBackgroundCycle()
            }
            PushEventType.PROCESS_COMPLETED -> {
//                recordStats = false
//                binding.chartHolder.visibility = View.GONE
//                binding.llCycleDetails.visibility = View.GONE
//                binding.scrollArea.visibility = View.GONE
//                binding.tvFlCompleted.visibility = View.VISIBLE
//                showMessage("Training has been completed.")
            }
            PushEventType.TYPE_WORKER_STATUS -> {
//                sendWorkerStatusToServer()
//                updateWorkerStatus()
            }
        }

        AppPreferences.putBoolean(pushEvent.eventType.name, true)
    }

    private fun launchForegroundCycle() {
        if(trainingTask == null) {
            trainingTask = TrainingTask(
                syftConfig,
                BuildConfig.SYFT_AUTH_TOKEN,
                femnistDataRepository,
                dataSet
            )
        }
        compositeDisposable.clear()
        compositeDisposable.add(trainingTask!!.runTask(ServiceLogger()).subscribe())
    }

    inner class ServiceLogger : MnistLogger {

        override val logText
            get() = logTextInternal
        private val logTextInternal = MutableLiveData<String>()

        override val cycleText
            get() = cycleTextInternal
        private val cycleTextInternal = MutableLiveData<String>()

        override val steps
            get() = stepsInternal
        private val stepsInternal = MutableLiveData<String>()

        override val processState
            get() = processStateInternal
        private val processStateInternal = MutableLiveData<ContentState>()

        override val processData
            get() = processDataInternal
        private val processDataInternal = MutableLiveData<ProcessData>()

//        private val workManager = WorkManager.getInstance(serviceContext)

        override fun postState(status: ContentState) {
            publish(STATUS to status.toString())
        }

        override fun postData(result: Float) {
            publish(LOSS_LIST to result)
        }

        override fun postEpoch(epoch: String) {
//            if (epoch % 10 == 0 && getState() == WorkInfo.State.RUNNING)
//            setForegroundAsync(createForegroundInfo(epoch))
            startForeground(SERVICE_ID, createForegroundInfo(epoch))
            publish(EPOCH to epoch)
        }

        override fun postLog(message: String) {
            publish(LOG to message)
        }

        override fun postCycle(message: String) {
            publish(CYCLE to message)
        }

//        private fun getState() = workManager.getWorkInfoById(id).get().state

        private fun publish(pair: Pair<String, Any>) {
//            if (getState() == WorkInfo.State.RUNNING)
//                setProgressAsync(workDataOf(pair))
        }

    }

    @RequiresApi(Build.VERSION_CODES.O)
    private fun createChannel() {
        Timber.d("--------------Creating notification channel")
        val descriptionText = "testing"
        val importance = NotificationManager.IMPORTANCE_LOW
        val mChannel = NotificationChannel(CHANNEL_ID, CHANNEL_NAME, importance)
        mChannel.description = descriptionText
//        val notificationManager =
//            applicationContext.getSystemService(NOTIFICATION_SERVICE) as NotificationManager
//        notificationManager.createNotificationChannel(mChannel)

        val service = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        service.createNotificationChannel(mChannel)

    }

    private fun startFService(foregroundInfo: Notification) {

        startForeground(SERVICE_ID, foregroundInfo)
        IS_RUNNING = true
    }

    private fun stopService() {
        // Stop foreground service and remove the notification.
        stopForeground(true)
        // Stop the foreground service.
        stopSelf()

        IS_RUNNING = false
    }


    override fun onDestroy() {
        IS_RUNNING = false
        mHandler?.removeCallbacks(null)
    }

    companion object {

        const val TAG = "FOREGROUND_SERVICE"


        const val ACTION_STOP_FOREGROUND_SERVICE = "ACTION_STOP_FOREGROUND_SERVICE"

        const val ACTION_OPEN_APP = "ACTION_OPEN_APP"
        const val KEY_DATA = "KEY_DATA"

        private const val CHANNEL_ID: String = "1001"
        private const val CHANNEL_NAME: String = "Event Tracker"
        private const val SERVICE_ID: Int = 1
        private const val ONE_MIN_MILLI: Long = 1000  //1min

        var IS_RUNNING: Boolean = false
    }
}
