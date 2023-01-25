package org.openmined.syft.demo.federated.domain

import org.openmined.syft.demo.federated.datasource.LocalMNISTDataDataSource
import org.openmined.syft.demo.federated.datasource.LocalFEMNISTDataSource
import org.openmined.syft.execution.Plan
import org.pytorch.IValue
import org.pytorch.Tensor

class FEMNISTDataRepository constructor(
    private val localFEMNISTDataSource: LocalFEMNISTDataSource
) {
    fun loadDataBatch(batchSize: Int, plan: Plan): Pair<IValue, IValue> {
        val data = localFEMNISTDataSource.loadDataBatch(batchSize, plan)
        val tensorsX = IValue.from(Tensor.fromBlob(data.first.flattenedArray, data.first.shape))

        val tensorsY = IValue.from(Tensor.fromBlob(data.second.flattenedArray, data.second.shape))
        return Pair(tensorsX, tensorsY)
    }

    fun loadTestDataBatch(plan: Plan, forTrainData: Boolean): Pair<IValue, IValue> {
        val data = localFEMNISTDataSource.loadTestDataBatch(plan, forTrainData)
        val tensorsX = IValue.from(Tensor.fromBlob(data.first.flattenedArray, data.first.shape))

        val tensorsY = IValue.from(Tensor.fromBlob(data.second.flattenedArray, data.second.shape))
        return Pair(tensorsX, tensorsY)
    }

    fun getTotalNumberOfSamples():Pair<Int,Int> = localFEMNISTDataSource.getTotalNumberOfSamples()

    fun getIsAllBatchConsumed() = localFEMNISTDataSource.getIsAllBatchConsumed()

    fun setIsAllBatchConsumed(b: Boolean) {
        localFEMNISTDataSource.setIsAllBatchConsumed(b)
    }

    fun initDataReaders() {
        localFEMNISTDataSource.initDataReaders()
    }

    fun loadTestBatch(plan: Plan, forTrainData: Boolean): Pair<IValue, IValue> {
        return loadTestDataBatch(plan, forTrainData)
    }

    fun setIsRandomizationRequired(isRequired: Boolean) {
        localFEMNISTDataSource.setIsRandomizationRequired(isRequired)
    }

    fun setSeed(seed: Long) {
        localFEMNISTDataSource.setSeed(seed)
    }

    fun clearData() {
        localFEMNISTDataSource.clearAllData()
    }

    fun loadTestDataBatch(plan: Plan, oneHotVectorPlan: Plan, dataId: Int, forTrainData: Boolean = false): Pair<IValue, IValue> {
        val data = localFEMNISTDataSource.loadTestDataBatch(plan , oneHotVectorPlan, dataId, forTrainData)
        val tensorsX = IValue.from(Tensor.fromBlob(data.first.flattenedArray, data.first.shape))

        val tensorsY = IValue.from(Tensor.fromBlob(data.second.flattenedArray, data.second.shape))
        return Pair(tensorsX, tensorsY)
    }

    fun get_test_data_ids() = localFEMNISTDataSource.get_test_ids()
    fun get_train_data_ids() = localFEMNISTDataSource.get_train_ids()

    fun getTotalNumberOfSamples(trainDataId: Int, testDataId: Int) = localFEMNISTDataSource.getTotalNumberOfSamples(testDataId, trainDataId)
    fun is_slow_client() = localFEMNISTDataSource.is_slow_client()
}