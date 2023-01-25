package org.openmined.syft.proto

import android.util.Log
import org.openmined.syft.domain.SyftConfiguration
import org.openmined.syft.execution.Plan
import org.openmined.syftproto.execution.v1.StateOuterClass
import org.openmined.syftproto.execution.v1.StateTensorOuterClass
import org.pytorch.IValue
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

/** SyftState class is responsible for storing all the weights of the neural network.
 * We update these model weights after every plan.execute
 * @property placeholders the variables describing the location of tensor in the plan torchscript
 * @property iValueTensors the IValue tensors for the model params
 * @sample SyftModel.updateModel
 * @sample SyftModel.loadModelState
 */
@ExperimentalUnsignedTypes
data class SyftState(
    val placeholders: Array<Placeholder>,
    val iValueTensors: Array<IValue>
) {

    /**
     * @return an array of [SyftTensor] from the [State][[https://pytorch.org/javadoc/org/pytorch/IValue.html] IValue Array
     */
    val syftTensorArray get() = iValueTensors.map { it.toTensor().toSyftTensor() }.toTypedArray()

    /**
     * @return an array of pyTorch [Tensors][https://pytorch.org/javadoc/org/pytorch/Tensor.html] from the SyftTensors list
     */
    val tensorArray get() = iValueTensors.map { it.toTensor() }.toTypedArray()



    companion object {
        var oldSyftState: SyftState ? = null
        var epochSyftState: SyftState ? = null
        /**
         * Load the [SyftTensors][SyftTensor] and [placeholders][Placeholder] from the file
         */
        @ExperimentalUnsignedTypes
        fun loadSyftState(fileLocation: String): SyftState {
            return StateOuterClass.State.parseFrom(File(fileLocation).readBytes()).toSyftState()
        }

        fun createSyftState(iTensor: IValue, config: SyftConfiguration, plan: Plan): SyftState {
//            val activationFilePath = "${config.filesDir}/activations/activations.pb"

            print("============== createSyftState")
            val placeholders = arrayOf(Placeholder("1", listOf("1", "#state-1"))) // placeholders of 1 hidden layer
            val iValueTensors = arrayOf(iTensor) // activations of 1 hidden layer

            var newSyftState = SyftState(placeholders, iValueTensors)
            var serializedState: StateOuterClass.State? = null
//            var activationsFile = File(activationFilePath)
            if(oldSyftState != null) {
//                print("\nOLD ACTITVATIONS")
//                val oldSyftState = loadSyftState(activationFilePath)
//
//                printIValue(oldSyftState!!.iValueTensors[0])
//
//                print("\nNEW ACTITVATIONS")
//                printIValue(newSyftState.iValueTensors[0])

                val output = plan.execute(oldSyftState!!.iValueTensors[0], newSyftState.iValueTensors[0]) // activations of 1 hidden layer
                newSyftState = SyftState(placeholders, arrayOf(output!!))

//                print("\nSUMMED ACTITVATIONS")
//                printIValue(newSyftState.iValueTensors[0])

                oldSyftState = newSyftState
            } else {
//                if(!activationsFile.parentFile!!.exists(oldSyftState!!.iValueTensors[0]))
//                    activationsFile.parentFile!!.mkdirs()


                serializedState = newSyftState.serialize()
                oldSyftState = newSyftState
//                print("\nNew state arrived. oldState = ${oldSyftState == null}")
//                printIValue(newSyftState.iValueTensors[0])
            }

//            val outputStream = FileOutputStream(activationFilePath)
//            outputStream.write(serializedState!!.toByteArray())
//            outputStream.close()
            return newSyftState
        }

        fun performAvgOverSamples(avgPlan: Plan, sumPlan: Plan, count: Int) {
            val placeholders = arrayOf(Placeholder("1", listOf("1", "#state-1"))) // placeholders of 1 hidden layer
            val avgOutPut = avgPlan.execute(oldSyftState!!.iValueTensors[0], IValue.from(Tensor.fromBlob(longArrayOf(count.toLong()), longArrayOf(1))))
            if(epochSyftState != null) {
                // here we are adding activations of multiple epochs
                val output = sumPlan.execute(epochSyftState!!.iValueTensors[0], avgOutPut!!) // sum of epoch 1 tensors with averaged activations (avg over all samples)
                var newSyftState = SyftState(placeholders, arrayOf(output!!))
                epochSyftState = newSyftState
            } else {
                epochSyftState = SyftState(placeholders, arrayOf(avgOutPut!!))
            }

            oldSyftState = null
        }

        fun performAvgOverEpochs(avgPlan: Plan, count: Int): SyftState {
            val placeholders = arrayOf(Placeholder("1", listOf("1", "#state-1"))) // placeholders of 1 hidden layer
            val avgOutPut = avgPlan.execute(epochSyftState!!.iValueTensors[0], IValue.from(Tensor.fromBlob(longArrayOf(count.toLong()), longArrayOf(1))))

            return SyftState(placeholders, arrayOf(avgOutPut!!))
        }

        fun printIValue(iValue: IValue) {
            println()
            for(item in iValue.toTensor().dataAsFloatArray) {
                print(" $item ")
            }
            println()
        }

        fun resetOldSyftState() {
            oldSyftState = null
            epochSyftState = null
        }

    }

    /**
     * This method is used to save/update SyftState parameters.
     * @throws IllegalArgumentException if the size newModelParams is not correct.
     * @param newStateTensors a list of PyTorch Tensor that would be converted to syftTensor
     */
    fun updateState(newStateTensors: Array<IValue>) {
        if (iValueTensors.size != newStateTensors.size) {
            throw IllegalArgumentException("The size of the list of new parameters ${newStateTensors.size} is different than the list of params of the model ${iValueTensors.size}")
        }
        newStateTensors.forEachIndexed { idx, value ->
            iValueTensors[idx] = value
        }
    }

    /**
     * Subtract the older state from the current state to generate the diff
     * @param oldSyftState The state with respect to which the diff will be generated
     * @param diffScriptLocation The location of the torchscript for performing the subtraction
     * @throws IllegalArgumentException if the size newModelParams is not same.
     */
    fun createDiff(oldSyftState: SyftState, diffScriptLocation: String): SyftState {
        if (this.iValueTensors.size != oldSyftState.iValueTensors.size)
            throw IllegalArgumentException("Dimension mismatch. Original model params have size ${oldSyftState.iValueTensors.size} while input size is ${this.iValueTensors.size}")
        val diff = Array(size = this.iValueTensors.size) { index ->
            this.iValueTensors[index].applyOperation(
                diffScriptLocation,
                oldSyftState.iValueTensors[index]
            )
        }
        val localPlaceHolders = diff.mapIndexed { idx, _ ->
            Placeholder(
                idx.toString(),
                listOf("$idx", "#state-$idx")
            )
        }.toTypedArray()
        return SyftState(placeholders = localPlaceHolders, iValueTensors = diff)
    }

    fun createWeightedDiff(oldSyftState: SyftState, diffScriptLocation: String, weight: Int): SyftState {
        if (this.iValueTensors.size != oldSyftState.iValueTensors.size)
            throw IllegalArgumentException("Dimension mismatch. Original model params have size ${oldSyftState.iValueTensors.size} while input size is ${this.iValueTensors.size}")
        val diff = Array(size = this.iValueTensors.size) { index ->
//            var weightedTensor = this.iValueTensors[index].toTensor() * weight

            val originalTensor = this.iValueTensors[index].toTensor()
            val originalTensorFloatValues = originalTensor.dataAsFloatArray
            val weightedFloats  = arrayListOf<Float>()

            for (item in originalTensorFloatValues) {
                weightedFloats.add(item*weight)
            }

            val weightedITensorValue = IValue.from(Tensor.fromBlob(weightedFloats.toFloatArray(), originalTensor.shape()))


            weightedITensorValue.applyOperation(
                diffScriptLocation,
                oldSyftState.iValueTensors[index]
            )
        }

        val floatArray = diff[0].toTensor().dataAsFloatArray
        Log.d("Syft", "diff ${floatArray.toTypedArray()}")

        val localPlaceHolders = diff.mapIndexed { idx, _ ->
            Placeholder(
                idx.toString(),
                listOf("$idx", "#state-$idx")
            )
        }.toTypedArray()
        return SyftState(placeholders = localPlaceHolders, iValueTensors = diff)
    }

    /**
     * Generate StateOuterClass.State object using Placeholders list and syftTensor list
     */
    fun serialize(): StateOuterClass.State {
        return StateOuterClass.State.newBuilder().addAllPlaceholders(
            placeholders.map { it.serialize() }
        ).addAllTensors(syftTensorArray.map {
            StateTensorOuterClass.StateTensor
                    .newBuilder()
                    .setTorchTensor(it.serialize())
                    .build()
        }).build()
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as SyftState

        if (!placeholders.contentEquals(other.placeholders)) return false
        if (!iValueTensors.contentEquals(other.iValueTensors)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = placeholders.contentHashCode()
        result = 31 * result + iValueTensors.contentHashCode()
        return result
    }

    fun updateWeightedState(weight: Int) {

        iValueTensors.forEachIndexed { idx, value ->
            val originalTensorFloatValues = value.toTensor().dataAsFloatArray
            val weightedFloats  = arrayListOf<Float>()

            for (item in originalTensorFloatValues) {
                weightedFloats.add(item*weight)
            }

            val weightedITensorValue = IValue.from(Tensor.fromBlob(weightedFloats.toFloatArray(), value.toTensor().shape()))
            iValueTensors[idx] = weightedITensorValue
        }
    }
}

/**
 * Generate State object from StateOuterClass.State object
 */
@ExperimentalUnsignedTypes
fun StateOuterClass.State.toSyftState(): SyftState {
    val placeholders = this.placeholdersList.map {
        Placeholder.deserialize(it)
    }.toTypedArray()
    val syftTensors = this.tensorsList.map {
        it.torchTensor.toSyftTensor().getIValue()
    }.toTypedArray()
    return SyftState(placeholders, syftTensors)
}