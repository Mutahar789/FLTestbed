package org.openmined.syft.demo.federated.domain

import org.openmined.syft.demo.federated.datasource.LocalMNISTDataDataSource
import org.openmined.syft.demo.federated.datasource.LocalSyntheticDataSource
import org.openmined.syft.execution.Plan
import org.pytorch.IValue
import org.pytorch.Tensor

class SyntheticDataRepository constructor(
    private val localSyntheticDataSource: LocalSyntheticDataSource
) {
    fun loadDataBatch(batchSize: Int, plan: Plan): Pair<IValue, IValue> {
        val data = localSyntheticDataSource.loadDataBatch(batchSize, plan)
        val tensorsX = IValue.from(Tensor.fromBlob(data.first.flattenedArray, data.first.shape))

        val tensorsY = IValue.from(Tensor.fromBlob(data.second.flattenedArray, data.second.shape))
        return Pair(tensorsX, tensorsY)
    }

    fun loadTestDataBatch(plan: Plan, forTrainData: Boolean): Pair<IValue, IValue> {
        val data = localSyntheticDataSource.loadTestDataBatch(plan, forTrainData)
        val tensorsX = IValue.from(Tensor.fromBlob(data.first.flattenedArray, data.first.shape))

        val tensorsY = IValue.from(Tensor.fromBlob(data.second.flattenedArray, data.second.shape))
        return Pair(tensorsX, tensorsY)
    }

    fun getTotalNumberOfSamples():Pair<Int,Int> = localSyntheticDataSource.getTotalNumberOfSamples()

    fun getIsAllBatchConsumed() = localSyntheticDataSource.getIsAllBatchConsumed()

    fun setIsAllBatchConsumed(b: Boolean) {
        localSyntheticDataSource.setIsAllBatchConsumed(b)
    }

    fun loadTestBatch(plan: Plan, forTrainData: Boolean): Pair<IValue, IValue> {
        return loadTestDataBatch(plan, forTrainData)
    }

    fun setIsRandomizationRequired(isRequired: Boolean) {
        localSyntheticDataSource.setIsRandomizationRequired(isRequired)
    }

    fun setSeed(seed: Long) {
        localSyntheticDataSource.setSeed(seed)
    }
}