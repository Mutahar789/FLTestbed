package org.openmined.syft.demo.federated.datasource

import android.content.res.Resources
import org.json.JSONArray
import org.json.JSONObject
import org.openmined.syft.demo.R
import org.openmined.syft.demo.federated.domain.Batch
import org.openmined.syft.execution.Plan
import org.pytorch.IValue
import org.pytorch.Tensor
import timber.log.Timber
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.text.DecimalFormat
import kotlin.random.Random

class LocalFEMNISTDataSource constructor(
    private val resources: Resources
) {
    private var isSlowClient = false
    private var client = 0

    private lateinit var trainDataReader: JSONObject
    private lateinit var testDataReader: JSONObject
    
    private var train_data = arrayListOf<List<Float>>()
    private val CLASSES = 62
    private val SHAPE_SIZE = 784
    private var currentBatch = 0

    private var seed = 1549786796.toLong()
    private var isAllBatchConsumed = false
    private var isRandomizationRequired = false

    val trainInput = arrayListOf<ArrayList<Float>>()
    val label_categories = arrayListOf<Long>()

    // 0 = 51 samples
    // 1 = 19 samples

    lateinit var train_data_array:JSONArray
    lateinit var train_data_labels:JSONArray


    val testInput = arrayListOf<ArrayList<Float>>()

    /**
     * Batch wise data loading for training
     *
     */
    fun loadDataBatch(batchSize: Int, plan: Plan): Pair<Batch, Batch> {
        var startIndex = currentBatch * batchSize
        if(startIndex >= train_data_array.length()) {
            startIndex = 0
            currentBatch = 0
        }


        var batchEndIndex = startIndex + batchSize

        if(batchEndIndex >= train_data_array.length()) {
            batchEndIndex = train_data_array.length()
            isAllBatchConsumed = true
        }

        if(isRandomizationRequired) {
            val df = DecimalFormat("#.######")

            trainInput.clear()
            label_categories.clear()

            for (idx in 0 until train_data_array.length()) {

                val train_data = train_data_array.getJSONArray(idx)

                val train_float_list = arrayListOf<Float>()
                for (idx2 in 0 until train_data.length()) {
//                train_float_list.add((train_data[idx2] as Double).toFloat())
                    if (train_data[idx2] is Integer) {
                        train_float_list.add(
                            df.format((train_data[idx2] as Integer).toFloat()).toFloat()
                        )
                    }
                    else {
                        train_float_list.add(
                            df.format((train_data[idx2] as Double).toFloat()).toFloat()
                        )
                    }
                }

                trainInput.add(train_float_list)
            }

            for (idx in 0 until train_data_labels.length()) {
                label_categories.add(train_data_labels[idx].toString().toLong())
            }

//            trainInput.shuffle(Random(randomSeed))
//            label_categories.shuffle(Random(randomSeed))

//            shuffle(trainInput, trainInput.size, seed)
//            shuffleLabels(label_categories, label_categories.size, seed)

            isRandomizationRequired = false

//            Timber.d("===================== label_categories ${label_categories}")
//            Timber.d("\n\n===================== trainInput ${trainInput}\n\n")
        }



        currentBatch++

        val randomizedInputBatch = trainInput.subList(startIndex, batchEndIndex)
        val randomizedLabelBatch = label_categories.subList(startIndex, batchEndIndex)
//
//        Timber.d("=========================== randomizedInputBatch ${randomizedInputBatch}")
//        Timber.d("=========================== randomizedLabelBatch ${randomizedLabelBatch}")

        val one_hot_result_tensor = plan.execute(IValue.from(Tensor.fromBlob(randomizedLabelBatch.toLongArray(), longArrayOf(randomizedLabelBatch.size.toLong()))))?.toTensor()
        val one_hot_result = one_hot_result_tensor?.dataAsLongArray


        val one_hot_flattened_float_list = arrayListOf<Float>()
        one_hot_result?.let {
            for (item in one_hot_result) {
                one_hot_flattened_float_list.add(item.toFloat())
            }
        }

        val trainingData = Batch(
            randomizedInputBatch.flatten().toFloatArray(),
            longArrayOf(randomizedInputBatch.size.toLong(), SHAPE_SIZE.toLong())
        )

        val trainingLabel = Batch(
            one_hot_flattened_float_list.toFloatArray(),
            longArrayOf(randomizedLabelBatch.size.toLong(), CLASSES.toLong())
        )


        return Pair(trainingData, trainingLabel)
    }

    fun setIsAllBatchConsumed(isConsumed: Boolean) {
        this.isAllBatchConsumed = isConsumed
    }

    fun getIsAllBatchConsumed() = isAllBatchConsumed

    fun clearAllData() {
        trainInput.clear()
        label_categories.clear()
    }


    /**
     * Data will be loaded based on training / testing flag for evaluation purpose.
     */
    fun loadTestDataBatch(plan: Plan, forTrainData: Boolean = false): Pair<Batch, Batch> {

        // 0 = 51 samples
        // 1 = 19 samples
        var test_data_array:JSONArray? = JSONArray()
        var test_data_labels:JSONArray? = JSONArray()

        if(forTrainData) {
            test_data_array = trainDataReader!!.getJSONArray("x")
            test_data_labels = trainDataReader!!.getJSONArray("y")
        } else {
            test_data_array = testDataReader!!.getJSONArray("x")
            test_data_labels = testDataReader!!.getJSONArray("y")
        }

        testInput.clear()

        for (idx in 0 until test_data_array.length()) {
            val train_data = test_data_array.getJSONArray(idx)

            val train_float_list = arrayListOf<Float>()
            for (idx2 in 0 until train_data.length()){
                // Timber.d("Val: ${train_data[idx2]}")
                if (train_data[idx2] is Integer) {
                    train_float_list.add((train_data[idx2] as Integer).toFloat())
                }
                else {
                    train_float_list.add((train_data[idx2] as Double).toFloat())
                }
            }
            testInput.add(train_float_list)
        }

        var label_categories = arrayListOf<Long>()
        for (idx in 0 until test_data_labels.length()) {
            label_categories.add(test_data_labels[idx].toString().toLong())
        }

        val one_hot_result_tensor = plan.execute(IValue.from(Tensor.fromBlob(label_categories.toLongArray(), longArrayOf(label_categories.size.toLong()))))?.toTensor()
        val one_hot_result = one_hot_result_tensor?.dataAsLongArray


        val one_hot_flattened_float_list = arrayListOf<Float>()
        one_hot_result?.let {
            for (item in one_hot_result) {
                one_hot_flattened_float_list.add(item.toFloat())
            }
        }

        val testingData = Batch(
            testInput.flatten().toFloatArray(),
            longArrayOf(testInput.size.toLong(), SHAPE_SIZE.toLong())
        )

        val trainingLabel = Batch(
            one_hot_flattened_float_list.toFloatArray(),
            longArrayOf(label_categories.size.toLong(), CLASSES.toLong())
        )

        test_data_array = null
        test_data_labels = null
        label_categories.clear()
        testInput.clear()
        return Pair(testingData, trainingLabel)
    }

    private fun returnTrainDataReader(dataId: Int = -1): JSONObject? {
        var fileName = -1

//        if(isSlowClient) {
//            fileName = R.raw.slow_client_train_500 // 126 samples
//        } else {
//            fileName = R.raw.fast_client_train_1500 // 149 samples
//        }

        // IID
//        if(isSlowClient) {
//            fileName = R.raw.slow_client_train_500 // 500 samples
//        } else {
//            fileName = R.raw.slow_client_train_2_500 // 500 samples
//        }

        // Non iid
//        if(isSlowClient) {
//            fileName = R.raw.client_niid_1 // 500 samples
//        } else {
//            fileName = R.raw.client_niid_2 // 500 samples
//        }


        if(dataId == -1) {
            // Non iid
//            if (isSlowClient) {
//
//                fileName = R.raw.custom_niid_250_1 // 250 samples
//            } else {
//                fileName = R.raw.custom_niid_250_2 // 250 samples
//            }
        } else {
            fileName = dataId
        }

//        fileName = R.raw.sample

        val reader = BufferedReader(
            InputStreamReader(
                resources.openRawResource(fileName)
            )
        )

        try {
            val jsonString = reader.use { it.readText() }
            return JSONObject(jsonString)
        } catch (ioException: IOException) {
            ioException.printStackTrace()
            return null
        }
    }

    private fun returnTestDataReader(dataId: Int = -1): JSONObject? {

        var fileName = -1

//        if(isSlowClient) {
//            fileName = R.raw.slow_client_test_150 // 14 samples
//        } else {
//            fileName = R.raw.fast_client_test_250 // 17 samples
//        }

        // iid
//        if(isSlowClient) {
//            fileName = R.raw.slow_client_test_150 // 14 samples
//        } else {
//            fileName = R.raw.slow_client_test_2_150 // 17 samples
//        }

        // Non iid
//        if(isSlowClient) {
//            fileName = R.raw.client_niid_1_test // 150 samples
//        } else {
//            fileName = R.raw.client_niid_2_test // 500 samples
//        }

        if(dataId == -1) {
            // Non iid
//            if (isSlowClient) {
//                fileName = R.raw.custom_niid_test_250_1 // 250 samples
//            } else {
//                fileName = R.raw.custom_niid_test_250_2 // 250 samples
//            }
        } else {
            fileName = dataId
        }
//        fileName = R.raw.sample_test

        val reader = BufferedReader(
            InputStreamReader(
                resources.openRawResource(fileName)
            )
        )

        try {
            val jsonString = reader.use { it.readText() }
            return JSONObject(jsonString)
        } catch (ioException: IOException) {
            ioException.printStackTrace()
            return null
        }
    }

    fun get_train_ids(): ArrayList<Int> {
        val data_ids: ArrayList<Int> = arrayListOf()
        when (client) {
            0 -> data_ids.add(R.raw.train_0)
            1 -> data_ids.add(R.raw.train_1)
            2 -> data_ids.add(R.raw.train_2)
            3 -> data_ids.add(R.raw.train_3)
        }
        return data_ids
    }

    fun get_test_ids(): ArrayList<Int> {
        val data_ids: ArrayList<Int> = arrayListOf()

        when (client) {
            0 -> data_ids.add(R.raw.test_0)
            1 -> data_ids.add(R.raw.test_1)
            2 -> data_ids.add(R.raw.test_2)
            3 -> data_ids.add(R.raw.test_3)
        }

        return data_ids
    }

    fun getTotalNumberOfSamples(): Pair<Int, Int> {
        val test_data_labels = testDataReader!!.getJSONArray("y")
        val train_data_labels = trainDataReader!!.getJSONArray("y")

        return Pair(train_data_labels.length(), test_data_labels.length())
    }

    fun getTotalNumberOfSamples(testDataId: Int, trainDataId: Int): Pair<Int, Int> {

        var reader = BufferedReader(
                InputStreamReader(
                    resources.openRawResource(testDataId)
                )
                )
        var testDataReader = JSONObject(reader.use { it.readText() })
        reader = BufferedReader(
            InputStreamReader(
                resources.openRawResource(trainDataId)
            )
        )
        var trainDataReader = JSONObject(reader.use { it.readText() })


        val test_data_labels = testDataReader.getJSONArray("y")
        val train_data_labels = trainDataReader.getJSONArray("y")

        return Pair(train_data_labels.length(), test_data_labels.length())
    }

    fun setIsRandomizationRequired(isRequired: Boolean) {
        this.isRandomizationRequired = isRequired
    }

    fun setSeed(seed: Long) {
        this.seed = seed
    }

    fun loadTestDataBatch(plan: Plan, oneHotVectorPlan: Plan, dataId: Int, forTrainData: Boolean = false): Pair<Batch, Batch> {

        // 0 = 51 samples
        // 1 = 19 samples
        var test_data_array = JSONArray()
        var test_data_labels = JSONArray()

        val reader = BufferedReader(
            InputStreamReader(
                resources.openRawResource(dataId)
            )
        )
        var dataReader = JSONObject(reader.use { it.readText() })

        test_data_array = dataReader.getJSONArray("x")
        test_data_labels = dataReader.getJSONArray("y")

        testInput.clear()

        for (idx in 0 until test_data_array.length()) {
            val train_data = test_data_array.getJSONArray(idx)

            val train_float_list = arrayListOf<Float>()
            for (idx2 in 0 until train_data.length()){
                // Timber.d("Val: ${train_data[idx2]}")
                if (train_data[idx2] is Integer) {
                    train_float_list.add((train_data[idx2] as Integer).toFloat())
                }
                else {
                    train_float_list.add((train_data[idx2] as Double).toFloat())
                }
            }
            testInput.add(train_float_list)
        }

        var label_categories = arrayListOf<Long>()
        for (idx in 0 until test_data_labels.length()) {
            label_categories.add(test_data_labels[idx].toString().toLong())
        }

        val one_hot_result_tensor = oneHotVectorPlan.execute(IValue.from(Tensor.fromBlob(label_categories.toLongArray(), longArrayOf(label_categories.size.toLong()))))?.toTensor()
        val one_hot_result = one_hot_result_tensor?.dataAsLongArray


        val one_hot_flattened_float_list = arrayListOf<Float>()
        one_hot_result?.let {
            for (item in one_hot_result) {
                one_hot_flattened_float_list.add(item.toFloat())
            }
        }

        val testingData = Batch(
            testInput.flatten().toFloatArray(),
            longArrayOf(testInput.size.toLong(), SHAPE_SIZE.toLong())
        )

        val trainingLabel = Batch(
            one_hot_flattened_float_list.toFloatArray(),
            longArrayOf(label_categories.size.toLong(), CLASSES.toLong())
        )

        label_categories.clear()
        testInput.clear()
        return Pair(testingData, trainingLabel)
    }

    fun initDataReaders() {
        val trainIds = get_train_ids()
        val testIds = get_test_ids()

        val randomIndex = 0//Random.nextInt(trainIds.size);


//        Timber.e("------------------- Selecting client ${randomIndex + 1}")
        trainDataReader = returnTrainDataReader(trainIds[randomIndex])!!
        testDataReader = returnTestDataReader(testIds[randomIndex])!!

        train_data_array = trainDataReader!!.getJSONArray("x")
        train_data_labels = trainDataReader!!.getJSONArray("y")
    }

    fun is_slow_client() = isSlowClient
}