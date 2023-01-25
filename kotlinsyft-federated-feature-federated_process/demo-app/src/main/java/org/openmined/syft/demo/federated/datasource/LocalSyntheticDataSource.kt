package org.openmined.syft.demo.federated.datasource

import android.content.res.Resources
import org.json.JSONArray
import org.json.JSONObject
import org.openmined.syft.demo.R
import org.openmined.syft.demo.federated.domain.Batch
import org.openmined.syft.execution.Plan
import org.pytorch.IValue
import org.pytorch.Tensor
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.text.DecimalFormat

class LocalSyntheticDataSource constructor(
    private val resources: Resources
) {
    private var trainDataReader = returnTrainDataReader()
    private var testDataReader = returnTestDataReader()

    private var train_data = arrayListOf<List<Float>>()
    private val CLASSES = 5
    private val SHAPE_SIZE = 100
    private var currentBatch = 0

    private val USER_ID = "0"
    private var seed = 1549786796.toLong()
    private var isAllBatchConsumed = false
    private var isRandomizationRequired = false

    val trainInput = arrayListOf<ArrayList<Float>>()
    val label_categories = arrayListOf<Long>()

    /**
     * Batch wise data loading for training
     *
     */
    fun loadDataBatch(batchSize: Int, plan: Plan): Pair<Batch, Batch> {

        // 0 = 51 samples
        // 1 = 19 samples
        val train_data_array = trainDataReader!!.getJSONObject("user_data").getJSONObject(USER_ID).getJSONArray("x")
        val train_data_labels = trainDataReader!!.getJSONObject("user_data").getJSONObject(USER_ID).getJSONArray("y")

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
                for (idx2 in 0 until train_data.length())
                    train_float_list.add((train_data[idx2] as Double).toFloat())
//                    train_float_list.add(df.format((train_data[idx2] as Double).toFloat()).toFloat())

                trainInput.add(train_float_list)
            }

            for (idx in 0 until train_data_labels.length()) {
                label_categories.add(train_data_labels[idx].toString().toLong())
            }

//            trainInput.shuffle(Random(randomSeed))
//            label_categories.shuffle(Random(randomSeed))

            // Custom Shufling algorithm
//            shuffle(trainInput, trainInput.size, seed)
//            shuffleLabels(label_categories, label_categories.size, seed)

            isRandomizationRequired = false
        }



        currentBatch++

        val randomizedInputBatch = trainInput.subList(startIndex, batchEndIndex)
        val randomizedLabelBatch = label_categories.subList(startIndex, batchEndIndex)

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


    /**
     * Data will be loaded based on training / testing flag for evaluation purpose.
     */
    fun loadTestDataBatch(plan: Plan, forTrainData: Boolean = false): Pair<Batch, Batch> {

        // 0 = 51 samples
        // 1 = 19 samples
        val trainInput = arrayListOf<ArrayList<Float>>()
        var test_data_array = JSONArray()
        var test_data_labels = JSONArray()

        if(forTrainData) {
            test_data_array = trainDataReader!!.getJSONObject("user_data").getJSONObject(USER_ID)
                    .getJSONArray("x")
            test_data_labels = trainDataReader!!.getJSONObject("user_data").getJSONObject(USER_ID)
                    .getJSONArray("y")
        } else {
            test_data_array = testDataReader!!.getJSONObject("user_data").getJSONObject(USER_ID)
                    .getJSONArray("x")
            test_data_labels = testDataReader!!.getJSONObject("user_data").getJSONObject(USER_ID)
                    .getJSONArray("y")
        }

        for (idx in 0 until test_data_array.length()) {
            val train_data = test_data_array.getJSONArray(idx)

            val train_float_list = arrayListOf<Float>()
            for (idx2 in 0 until train_data.length())
                train_float_list.add((train_data[idx2] as Double).toFloat())

            trainInput.add(train_float_list)
        }

        val label_categories = arrayListOf<Long>()
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

        val trainingData = Batch(
            trainInput.flatten().toFloatArray(),
            longArrayOf(trainInput.size.toLong(), SHAPE_SIZE.toLong())
        )

        val trainingLabel = Batch(
            one_hot_flattened_float_list.toFloatArray(),
            longArrayOf(label_categories.size.toLong(), CLASSES.toLong())
        )
        return Pair(trainingData, trainingLabel)
    }

    private fun returnTrainDataReader(): JSONObject? {
        val reader = BufferedReader(
            InputStreamReader(
                resources.openRawResource(R.raw.synthetic_test_1000)
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

    private fun returnTestDataReader(): JSONObject? {
        val reader = BufferedReader(
            InputStreamReader(
                resources.openRawResource(R.raw.synthetic_test_1000)
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

    fun getTotalNumberOfSamples(): Pair<Int, Int> {
        val test_data_labels = testDataReader!!.getJSONObject("user_data").getJSONObject(USER_ID).getJSONArray("y")
        val train_data_labels = trainDataReader!!.getJSONObject("user_data").getJSONObject(USER_ID).getJSONArray("y")

        return Pair(train_data_labels.length(), test_data_labels.length())
    }

    fun setIsRandomizationRequired(isRequired: Boolean) {
        this.isRandomizationRequired = isRequired
    }

    fun setSeed(seed: Long) {
        this.seed = seed
    }
}