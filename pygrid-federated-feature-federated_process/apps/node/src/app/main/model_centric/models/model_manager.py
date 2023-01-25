# PyGrid imports
# Syft dependencies
import syft as sy
from syft.execution.placeholder import PlaceHolder
from syft.execution.state import State
from syft.serde import protobuf
from syft_proto.execution.v1.state_pb2 import State as StatePB

from ...core.exceptions import ModelNotFoundError
from ...core.warehouse import Warehouse
from ..models.ai_model import Model, ModelCheckPoint
from ..models.model_mask import ModelMask
import numpy as np
import torch as th
import math
import random
import logging
th.set_printoptions(precision=8)
from torch import nn

class ModelManager:
    def __init__(self):
        self._models = Warehouse(Model)
        self._model_checkpoints = Warehouse(ModelCheckPoint)
        self.model_mask = Warehouse(ModelMask)
        self.DATASET = "FEMNIST"

    def create(self, model, process, mask):
        # Register new model
        _model_obj = self._models.register(flprocess=process)

        # Save model initial weights into ModelCheckpoint
        self._model_checkpoints.register(
            value=model, model=_model_obj, number=1, alias="latest"
        )

        self.model_mask.register(model_id=_model_obj.id, mask=mask)

        return _model_obj

    def update_mask(self, model_id, new_mask):
        old_mask_object = self.model_mask.first(model_id=model_id)
        
        if self.DATASET == "FEMNIST":
            old_masks = old_mask_object.mask.split(";")
            old_masks[2] = new_mask
            old_mask_object.mask = ';'.join(old_masks)
            print()
            print("====================UPDATED MASK", old_mask_object.mask, len(new_mask.split(',')))
        else:
            old_mask_object.mask = new_mask

        self.model_mask.update()

    def update_cnn_mask(self, model_id, new_mask):
        old_mask_object = self.model_mask.first(model_id=model_id)
        old_mask_object.mask = new_mask
        self.model_mask.update()

    def save(self, model_id: int, data: bin):
        """Create a new model checkpoint.

        Args:
            model_id: Model ID.
            data: Model data.
        Returns:
            model_checkpoint: ModelCheckpoint instance.
        """

        checkpoints_count = self._model_checkpoints.count(model_id=model_id)

        # Reset "latest" alias
        self._model_checkpoints.modify(
            {"model_id": model_id, "alias": "latest"}, {"alias": ""}
        )

        # Create new checkpoint
        new_checkpoint = self._model_checkpoints.register(
            model_id=model_id, value=data, number=checkpoints_count + 1, alias="latest"
        )
        return new_checkpoint

    def load(self, **kwargs):
        """Load model's Checkpoint."""
        _check_point = self._model_checkpoints.last(**kwargs)

        if not _check_point:
            raise ModelNotFoundError

        return _check_point

    def get_all(self, **kwargs):
        """Load model's Checkpoint."""
        _check_point = self._model_checkpoints.query(**kwargs)

        if not _check_point:
            raise ModelNotFoundError

        return _check_point

    def get(self, **kwargs):
        """Retrieve the model instance object.

        Args:
            process_id : Federated Learning Process ID attached to this model.
        Returns:
            model : SQL Model Object.
        Raises:
            ModelNotFoundError (PyGridError) : If model not found.
        """
        _model = self._models.last(**kwargs)

        if not _model:
            raise ModelNotFoundError

        return _model

    def get_mask(self, **kwargs):
        """Retrieve the model mask instance object.
        """
        _mask = self.model_mask.last(**kwargs)

        if not _mask:
            raise ModelNotFoundError

        return _mask

    @staticmethod
    def serialize_model_params(params):
        """Serializes list of tensors into State/protobuf."""
        model_params_state = State(
            state_placeholders=[PlaceHolder().instantiate(param) for param in params]
        )

        # make fake local worker for serialization
        worker = sy.VirtualWorker(hook=None)

        pb = protobuf.serde._bufferize(worker, model_params_state)
        serialized_state = pb.SerializeToString()

        return serialized_state

    @staticmethod
    def unserialize_model_params(bin: bin):
        """Unserializes model or checkpoint or diff stored in db to list of
        tensors."""
        state = StatePB()
        state.ParseFromString(bin)
        worker = sy.VirtualWorker(hook=None)
        state = protobuf.serde._unbufferize(worker, state)
        model_params = state.tensors()
        return model_params

    # def remove_neurons_layer(self, transponsed_weights, indices_to_remove, isoutput=None):
    #     remaining_neurons = np.delete(transponsed_weights, indices_to_remove, axis=1)
    #     if isoutput is None:
    #         return th.transpose(remaining_neurons, 0, 1)
    #     else:
    #         return remaining_neurons
    #
    # def remove_neurons_bias(self, bias, indices_to_remove):
    #     return np.delete(bias, indices_to_remove)
    #
    #
    # def insert_neurons_layer(self, transponsed_weights, indices_to_insert, isoutput=None):
    #     indices = indices_to_insert[0]
    #     prev_con = transponsed_weights
    #     for i in indices:
    #         if isoutput is None:
    #             prev_con = np.insert(prev_con, i, 0, axis=0)
    #         else:
    #             prev_con = np.insert(prev_con, i, 0, axis=1)
    #
    #     return prev_con
    #
    #
    # def insert_neurons_bias(self, bias, indices_to_add):
    #     prev_con = bias
    #     indices = indices_to_add[0]
    #     for i in indices:
    #         prev_con = np.insert(prev_con, i, 0, axis=0)
    #     return prev_con
    #
    # def prune_model(self, model_id):
    #     model_checkpoint = self.load(model_id=model_id)
    #     model_params = self.unserialize_model_params(model_checkpoint.value) # first hidden layer
    #
    #     first_hidden_layer_params = model_params[0]
    #     first_hidden_layer_bias = model_params[1]
    #     output_layer = model_params[2]
    #
    #     transponsed_weights = th.transpose(first_hidden_layer_params, 0, 1)
    #
    #     masks = self.get_mask(model_id=model_id)
    #     mask = self.convert_str_to_mask(masks.mask)
    #     indices_to_remove = np.where( mask == 0)
    #     # print(f"========================== pruning model indices_to_remove {indices_to_remove}, mask {mask}")
    #
    #     pruned_first_hidden_layer_params = self.remove_neurons_layer(transponsed_weights, indices_to_remove)
    #     pruned_first_hidden_layer_bias = self.remove_neurons_bias(first_hidden_layer_bias, indices_to_remove)
    #     pruned_output_layer = self.remove_neurons_layer(output_layer, indices_to_remove, True)
    #
    #     model_params[0] = pruned_first_hidden_layer_params
    #     model_params[1] = pruned_first_hidden_layer_bias
    #     model_params[2] = pruned_output_layer
    #
    #     return self.serialize_model_params(model_params)
    #
    # def convert_pruned_model_to_original(self, model_params):
    #     _model = self.get(fl_process_id=1)
    #     model_id = _model.id
    #
    #     first_hidden_layer_params = model_params[0]
    #     first_hidden_layer_bias = model_params[1]
    #     output_layer = model_params[2]
    #
    #     # transponsed_weights = th.transpose(first_hidden_layer_params, 0, 1)
    #
    #     masks = self.get_mask(model_id=model_id)
    #     mask = self.convert_str_to_mask(masks.mask)
    #     indices_to_add = np.where( mask == 0)
    #     # print(f"========================== before converting model_params {model_params}")
    #     # print(f"========================== before converting indices_to_add {indices_to_add}")
    #
    #     pruned_first_hidden_layer_params = self.insert_neurons_layer(first_hidden_layer_params, indices_to_add)
    #     pruned_first_hidden_layer_bias = self.insert_neurons_bias(first_hidden_layer_bias, indices_to_add)
    #     pruned_output_layer = self.insert_neurons_layer(output_layer, indices_to_add, True)
    #
    #     model_params[0] = pruned_first_hidden_layer_params
    #     model_params[1] = pruned_first_hidden_layer_bias
    #     model_params[2] = pruned_output_layer
    #
    #     # print(f"========================== after converting model_params {model_params}")
    #
    #     return self.serialize_model_params(model_params)
    #
    #
    def convert_pruned_activations_to_original(self, activation):
        _model = self.get(fl_process_id=1)
        model_id = _model.id
    
        masks = self.get_mask(model_id=model_id)
        mask = self.convert_str_to_mask(masks.mask)
        indices_to_add = np.where( mask == 0)
        # print(f"=================== indices_to_add {indices_to_add}, mask {mask}")
        # print(f"========================== after converting original_scaled_activation {activation}")
    
        original_scaled_activation = self.insert_neurons_bias(activation, indices_to_add)
    
        # print(f"========================== after converting original_scaled_activation {original_scaled_activation}")
        array = []
        array.append(original_scaled_activation)
    
        return self.serialize_model_params(array)
    #
    #
    # def create_mask(self, params, prune_percentage, seed):
    #     logging.info("Creating mask")
    #     transponsed_weights = th.transpose(params, 0, 1)
    #     neurons_shape = list(transponsed_weights.shape)
    #     masks = np.zeros(neurons_shape[1])
    #     number_of_neurons = len(masks)
    #
    #     random.seed(seed)
    #     np.random.seed(seed)
    #
    #     neurons_to_keep = math.ceil(number_of_neurons - (prune_percentage * number_of_neurons))
    #     random_indices = np.random.choice(number_of_neurons, size=neurons_to_keep, replace=False)
    #     values = [1] * len(random_indices)
    #
    #     np.put(masks, random_indices, values)
    #
    #     str_mask = ','.join([str(int(num)) for num in masks])
    #
    #
    #     # exit()
    #
    #     # seed = 1549775860
    #     # str_mask = "0,1,1,0,1,1,0,1,0,0,0,1,0,0,0,1,1,0,1,1,0,0,1,1,0,0,0,0,1,1,1,1;1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0;0,0,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,0,1,0,0,1,1,0,1,1,1,0,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,1,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,0,1,0,0,0,1,1,1,0,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,0,0,0,1,1,1,0,0,0,1,1,0,1,1,1,0,0,1,0,1,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,1,1,1,0,1,1,0,1,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,1,0,1,1,0,0,1,1,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,1,1,1,0,1,1,1,0,0,0,1,0,1,1,0,1,0,1,1,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,1,0,1,1,1,1,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,0,0,1,1,1,1,0,0,0,0,1,1,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,1,1,0,0,1,1,0,1,0,0,1,0,1,0,1,0,1,1,1,0,1,1,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,0,1,0,1,1,1,0,1,0,0,0,0,0,0,0,1,1,1,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,1,1,0,1,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,0,1,0,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,1,0,1,0,0,1,1,1,0,0,0,1,1,1,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,0,1,1,1,0,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,1,1,1,1,1,1,0,1,1,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1,1,1,0,1,0,1,0,1,1,1,1,0,0,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,1,0,0,1,0,1,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,1,0,1,1,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,1,0,0,0,1,1,0,0,1,0,0,1,0,1,0,0,1,1,1,1,0,1,1,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1,1,0,1,0,1,1,1,0,0,0,1,1,1,0,1,1,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,0,0,0,0,1,1,1,0,1,0,1,1,0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,0,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,1,1,0,0,0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,0,1,0,1,1,1,1,0,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,1,1,0,1,0,0,1,0,0,0,0,1,1,1,1,0,0,1,1,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,1,1,0,0,0,1,1,1,0,1,1,1,0,0,0,1,0,1,0,0,0,1,1,1,0,0,1,0,1,1,0,0,0,1,0,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,0,0,1,0,1,0,0,0,0,0,1,1,1,1,0,1,1,0,1,0,0,1,1,1,1,1,1,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,1,1,0,0,1,0,1,0,1,0,1,1,1,0,0,1,1,1,1,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,1,0,0,0,0,0,1,1,0,1,1,0,0,1,1,0,1,0,1,1,1,0,1,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,1,1,0,0,0,0,1,1,1,0,1,1,0,0,1,0,1,1,0,1,1,0,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,0,1,0,1,1,0,1,0,0,1,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,1,0,1,1,0,0,1,0,0,0,1,1,0,0,1,1,1,1,0,1,0,0,1,1,0,0,0,1,0,0,0,1,1,1,1,1,1,0,1,0,0,0,1,1,1,1,0,1,0,1,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,1,0,1,0,0,1,0,1,0,0,0,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,0,0,1,0,0,1,1,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,0,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,0,1,0,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,0,1,1,1,1,1,1,0,0,1,1,0,0,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,0,0,1,1,1,1,0,1,1,1,0,1,1,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,1,1,1,0,0,1,0,0,1,0,1,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,1,0,0,1,0"
    #
    #     return str_mask


    ## Removing neurons & filters
    def remove_neurons_layer(self, weights, indices_to_remove):
        return np.delete(weights, indices_to_remove, axis=0)

    def remove_neurons_next_layer(self, weights_next_layer, indices_to_remove):
        return np.delete(weights_next_layer, indices_to_remove, axis=1)

    def remove_filters_layer(self, filters, indices_to_remove):
        return np.delete(filters, indices_to_remove, axis=0)

    def remove_filters_next_layer(self, next_layer, indices_to_remove):
        return np.delete(next_layer, indices_to_remove, axis=1)

    def remove_neurons_bias(self, bias, indices_to_remove):
        return np.delete(bias, indices_to_remove)

    ## Inserting neurons & filters
    def insert_neurons_layer(self, weights, indices):
        new_weights = weights.copy() 
        for i in indices:
            new_weights = np.insert(new_weights, i, 0, axis=0)
        return new_weights

    def insert_neurons_next_layer(self, weights_next_layer, indices):
        new_weights = weights_next_layer.copy() 
        for i in indices:
            new_weights = np.insert(new_weights, i, 0, axis=1)
        return new_weights

    def insert_filters_layer(self, filters, indices):
        new_filters = filters.copy()
        for i in indices:
            new_filters = np.insert(new_filters, i, 0, axis=0)
        return new_filters

    def insert_filters_next_layer(self, filters, indices):
        new_filters = filters.copy()
        for i in indices:
            new_filters = np.insert(new_filters, i, 0, axis=1)
        return new_filters

    def insert_neurons_bias(self, bias, indices_to_add):
        prev_con = bias
        indices = indices_to_add
        for i in indices:
            prev_con = np.insert(prev_con, i, 0, axis=0)
        return prev_con


    def printModelDimensions(self, model):
        for layer in model:
            print(layer.shape)

    def prune_model(self, model_id, seed):
        random.seed(seed)
        np.random.seed(seed)

        model_checkpoint = self.load(model_id=model_id)
        model_params = self.unserialize_model_params(model_checkpoint.value)  # first hidden layer

        print("=========================================================================")

        layer_indexes = [0, 2, 4]
        bias_indexes = [1, 3, 5]

        str_masks = self.get_mask(model_id=model_id).mask.split(';')

        masks = list(map(lambda str_mask: self.convert_str_to_mask(str_mask), str_masks))

        print("PRUNED MASKS:", masks)

        for i, layer_index in enumerate(layer_indexes):
            mask = masks[i]
            indices_to_remove = np.where(np.array(mask) == 0)[0]

            if layer_index == 0:
                model_params[0] = self.remove_filters_layer(model_params[0], indices_to_remove)
                model_params[2] = self.remove_filters_next_layer(model_params[2], indices_to_remove)

            elif layer_index == 2:
                numFilters = model_params[2].shape[0]
                numNeuronsDense = model_params[4].shape[0]

                model_params[2] = self.remove_filters_layer(model_params[2], indices_to_remove)
                model_params[4] = model_params[4].reshape(-1, numFilters, 7, 7)

                model_params[4] = self.remove_filters_next_layer(model_params[4], indices_to_remove)
                model_params[4] = model_params[4].reshape(numNeuronsDense, -1)

            elif layer_index == 4:
                model_params[4] = self.remove_neurons_layer(model_params[4], indices_to_remove)

                model_params[6] = self.remove_neurons_next_layer(model_params[6], indices_to_remove)


            model_params[bias_indexes[i]] = self.remove_neurons_bias(model_params[bias_indexes[i]], indices_to_remove)


        print("=========================================================================")

        print()
        print()
        print()
        self.printModelDimensions(model_params)
        print()
        print()
        print()

        return self.serialize_model_params(model_params)

    def convert_pruned_model_to_original(self, model_params):
        print(
            "====================================CONVERTING TO ORIGNAL===========================================")
        _model = self.get(fl_process_id=1)
        model_id = _model.id

        # first_hidden_layer_params = model_params[0]
        # first_hidden_layer_bias = model_params[1]
        second_last_hidden_layer_params = model_params[-4]
        second_last_hidden_layer_bias = model_params[-3]
        output_layer = model_params[-2]

        # transponsed_weights = th.transpose(first_hidden_layer_params, 0, 1)
        ########################

        layer_indexes = [0, 2, 4]
        bias_indexes = [1, 3, 5]

        str_masks = self.get_mask(model_id=model_id).mask.split(';')

        masks = list(map(lambda str_mask: self.convert_str_to_mask(str_mask), str_masks))

        for i, layer_index in enumerate(layer_indexes):
            mask = masks[i]
            indices_to_add = np.where(np.array(mask) == 0)[0]

            if layer_index == 0:
                model_params[0] = self.insert_filters_layer(model_params[0], indices_to_add)
                model_params[2] = self.insert_filters_next_layer(model_params[2], indices_to_add)

            elif layer_index == 2:
                numFilters = model_params[2].shape[0]
                numNeuronsDense = model_params[4].shape[0]

                model_params[2] = self.insert_filters_layer(model_params[2], indices_to_add)
                model_params[4] = model_params[4].reshape(-1, numFilters, 7, 7)
                model_params[4] = self.insert_filters_next_layer(model_params[4], indices_to_add).reshape(numNeuronsDense, -1)

            elif layer_index == 4:
                model_params[4] = self.insert_neurons_layer(model_params[4], indices_to_add)
                model_params[6] = self.insert_neurons_next_layer(model_params[6], indices_to_add)


            model_params[bias_indexes[i]] = self.insert_neurons_bias(model_params[bias_indexes[i]], indices_to_add)


        print("=========================================================================")

        print()
        print()
        print()
        self.printModelDimensions(model_params)
        print()
        print()
        print()

        ########################

        return self.serialize_model_params(model_params)

    def convert_pruned_activations_to_original(self, activation):
        _model = self.get(fl_process_id=1)
        model_id = _model.id

        masks = self.get_mask(model_id=model_id)

        if self.DATASET == "FEMNIST":
            mask = self.convert_str_to_mask(masks.mask.split(';')[2])
        else:
            mask = self.convert_str_to_mask(masks.mask)

        indices_to_add = np.where(mask == 0)[0]
        # print(f"=================== indices_to_add {indices_to_add}, mask {mask}")
        # print(f"========================== after converting original_scaled_activation {activation}")

        original_scaled_activation = self.insert_neurons_bias(activation, indices_to_add)

        # print(
        #     f"========================== after converting original_scaled_activation {original_scaled_activation}")
        array = []
        array.append(original_scaled_activation)

        return self.serialize_model_params(array)

    def create_mask(self, params, seed, prune_percentage):
        str_mask = ""

        random.seed(seed)
        np.random.seed(seed)

        if not self.DATASET == "FEMNIST":
            transponsed_weights = th.transpose(params[0], 0, 1)
            neurons_shape = list(transponsed_weights.shape)
            masks = np.zeros(neurons_shape[1])
            number_of_neurons = len(masks)
            drop_rate = 0.5
            neurons_to_keep = math.ceil(number_of_neurons - (drop_rate * number_of_neurons))
            random_indices = np.random.choice(number_of_neurons, size=neurons_to_keep, replace=False)
            values = [1] * len(random_indices)

            np.put(masks, random_indices, values)

            str_mask = ','.join([str(int(num)) for num in masks])

        else:
            conv1_index = 0
            conv2_index = 2
            dense1_index = 4

            layer_indexes = [conv1_index, conv2_index, dense1_index]
            drop_rate = prune_percentage

            for layer_index in layer_indexes:
                if layer_index == 0 or layer_index == 2:
                    # filter_shape = list(params[layer_index].shape)
                    # number_of_filters = filter_shape[0]
                    # mask = np.zeros(number_of_filters)
                    # filters_to_keep = math.ceil(number_of_filters - (drop_rate * number_of_filters))
                    # random_indices = np.random.choice(number_of_filters, size=filters_to_keep, replace=False)
                    # values = [1] * len(random_indices)
                    #
                    # np.put(mask, random_indices, values)

                    N = list(params[layer_index].shape)[0]
                    m = np.ones(N, dtype=int)
                    dropN = int(N * drop_rate)
                    dropInd = np.random.choice(N, dropN, replace=False)
                    m[dropInd] = 0
                    # info[i].append(m)

                    retained = 0
                    for num in m:
                        if num == 1:
                            retained += 1
                    # print("retained nurons in layer ", layer_index, " count ", retained)

                    str_mask += ','.join([str(int(num)) for num in m])
                    str_mask += ";"

                else:
                    transposed_weights = th.transpose(params[layer_index], 0, 1)
                    neurons_shape = list(transposed_weights.shape)
                    number_of_neurons = neurons_shape[1]
                    # mask = np.zeros(number_of_neurons)
                    # neurons_to_keep = math.ceil(number_of_neurons - (drop_rate * number_of_neurons))
                    # random_indices = np.random.choice(number_of_neurons, size=neurons_to_keep, replace=False)
                    # values = [1] * len(random_indices)

                    N = number_of_neurons
                    m = np.ones(N, dtype=int)
                    dropN = int(N * drop_rate)
                    dropInd = np.random.choice(N, dropN, replace=False)
                    m[dropInd] = 0

                    # np.put(mask, random_indices, values)

                    # retained = 0
                    # for num in m:
                    #     if num == 1:
                    #         retained += 1
                    # print("retained nurons in layer ", layer_index, " count ", retained)

                    str_mask += ','.join([str(int(num)) for num in m])

        # print(str_mask)
        #
        # raise SystemExit(0)

        # str_mask = "0,0,1,1,1,0,0,0,0,1,0,1,1,0,1,0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,1,1;1,1,0,0,1,1,0,0,1,1,0,1,0,0,0,1,1,1,0,0,0,1,1,0,0,1,0,1,1,1,0,1,1,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,1,0,1,0,0,1,1,1,0,0,1,1,1;1,1,0,0,1,0,0,1,1,0,0,1,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,1,0,0,0,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,1,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,0,0,0,1,0,1,1,1,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,1,0,0,0,0,1,1,0,1,0,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,1,0,1,0,0,0,1,1,0,1,0,0,1,0,0,0,0,1,1,1,0,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,0,1,0,0,0,0,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,1,1,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,1,0,0,1,0,1,1,0,0,0,1,1,1,0,1,0,1,0,1,0,0,0,1,0,1,1,1,0,0,0,0,0,1,1,1,1,1,0,1,0,1,1,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,1,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,1,0,1,1,0,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,0,1,0,1,1,0,0,0,1,0,0,1,1,1,1,0,1,1,1,0,0,0,0,0,1,0,0,0,0,1,1,1,0,1,1,1,1,0,0,0,0,1,0,1,1,0,1,0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,1,0,0,1,0,0,0,0,0,1,1,0,1,1,1,0,0,0,1,0,0,0,1,0,1,1,0,0,1,1,1,1,0,0,0,1,1,1,0,1,0,0,1,0,0,0,1,0,1,0,0,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,0,1,0,0,0,1,0,1,1,0,1,1,1,1,0,1,1,1,1,0,1,0,0,0,1,1,0,1,1,1,1,0,1,1,0,0,1,0,0,1,0,1,0,1,0,1,1,0,1,0,1,0,0,0,0,1,0,1,1,1,1,1,0,0,0,1,0,1,0,0,1,0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,0,0,1,1,0,0,1,1,0,0,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,1,0,1,0,0,1,0,1,0,0,1,1,0,1,0,1,1,0,1,0,0,1,1,0,1,1,1,0,0,1,1,0,1,0,0,1,0,1,0,0,0,0,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,0,0,1,1,1,1,1,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,1,1,1,0,0,1,1,1,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0,0,1,0,1,0,0,1,0,1,1,0,0,1,1,1,1,1,0,1,0,1,0,1,1,0,0,0,1,1,0,0,0,0,1,0,1,1,1,1,0,0,0,1,0,1,0,0,1,0,0,1,0,0,1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,1,1,0,1,1,0,1,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,1,0,0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,1,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,0,0,1,0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,1,1,1,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,1,1,1,0,0,1,1,0,0,0,0,1,0,0,1,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,1,0,1,0,0,1,1,0,1,1,1,0,1,0,1,0,0,1,1,1,1,1,1,1,0,1,1,0,1,0,0,1,1,1,1,0,0,0,0,0,1,1,0,1,1,0,1,0,0,1,1,0,0,1,0,0,0,0,1,1,0,1,0,0,1,1,1,0,1,0,1,1,1,1,0,0,0,1,0,0,1,0,0,0,1,1,0,1,1,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,1,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1,1,1,0,1,1,0,0,1,1,0,0,1,0,1,1,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,1,1,0,1,0,0,0,0,1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,1,1,1,1,0,0,1,0,0,1,0,0,1,1,1,0,0,1,1,1,0,1,1,1,1,0,1,0,0,0,1,1,1,0,1,1,0,1,1,1,0,0,1,0,1,1,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,0,1,1,0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,1,1,0,0,0,1,1,1,0,1,1,1,0,1,0,1,0,1,1,0,1,1,0,0,1,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,1,1,1,1,1,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1,1,1,0,1,1,0,1,1,0,0,1,0,1,0,1,1,1,0,0,0,0,0,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,1,1,0,1,0,0,1,0,1,1,1,1,1,0,0,1,0,0,0,0,0,1,0,1,0,1,1,1,0,1,0,1,1,1,1,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,1,1,0,1,0,0,1,1,0,1,1,1,0,1,0,0,1,1,0,1,0,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,0,0,1,1,0,0,0,1,0,1,1,1,0,1,0,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,1,0,1,1,1,0,1,0,1,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,1,1,0,0,1,1,0,1,1,0,1,0,1,1,1,1,0,1,1,0,0,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,1,1,0,1,0,0,1,1,0,1,1,0,0,1,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,1,1,1,0,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,0,1,1,1,1,0,0,1,1,1,0,1,1,0,0,1,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,0,1,0,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,0,1,0,1,0,1,1,1,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,1,0,1"
        return str_mask

    def convert_str_to_mask(self, str_mask):
        return np.array([int(i) for i in str_mask.split(',')])

    def get_masks_leaf_type(self):
        _model = self.get(fl_process_id=1)
        model_id = _model.id
        mask = self.get_mask(model_id=model_id)
        # mask = "0,1,1,0,1,1,0,1,0,0,0,1,0,0,0,1,1,0,1,1,0,0,1,1,0,0,0,0,1,1,1,1;1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,0,1,0,1,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0;0,0,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,0,1,0,0,1,1,0,1,1,1,0,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,1,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,0,1,0,0,0,1,1,1,0,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,0,0,0,1,1,1,0,0,0,1,1,0,1,1,1,0,0,1,0,1,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,1,1,1,0,1,1,0,1,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,1,0,1,1,0,0,1,1,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,1,1,1,0,1,1,1,0,0,0,1,0,1,1,0,1,0,1,1,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,1,0,1,1,1,1,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,0,0,1,1,1,1,0,0,0,0,1,1,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,1,1,0,0,1,1,0,1,0,0,1,0,1,0,1,0,1,1,1,0,1,1,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,0,1,0,1,1,1,0,1,0,0,0,0,0,0,0,1,1,1,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,1,1,0,1,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,0,1,0,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,1,0,1,0,0,1,1,1,0,0,0,1,1,1,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,0,1,1,1,0,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,1,1,1,1,1,1,0,1,1,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1,1,1,0,1,0,1,0,1,1,1,1,0,0,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,1,0,0,1,0,1,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,1,0,1,1,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,1,0,0,0,1,1,0,0,1,0,0,1,0,1,0,0,1,1,1,1,0,1,1,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1,1,0,1,0,1,1,1,0,0,0,1,1,1,0,1,1,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,0,0,0,0,1,1,1,0,1,0,1,1,0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,0,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,1,1,0,0,0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,0,1,0,1,1,1,1,0,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,1,1,0,1,0,0,1,0,0,0,0,1,1,1,1,0,0,1,1,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,1,1,0,0,0,1,1,1,0,1,1,1,0,0,0,1,0,1,0,0,0,1,1,1,0,0,1,0,1,1,0,0,0,1,0,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,0,0,1,0,1,0,0,0,0,0,1,1,1,1,0,1,1,0,1,0,0,1,1,1,1,1,1,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,1,1,0,0,1,0,1,0,1,0,1,1,1,0,0,1,1,1,1,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,1,0,0,0,0,0,1,1,0,1,1,0,0,1,1,0,1,0,1,1,1,0,1,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,1,1,0,0,0,0,1,1,1,0,1,1,0,0,1,0,1,1,0,1,1,0,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,0,1,0,1,1,0,1,0,0,1,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,1,0,1,1,0,0,1,0,0,0,1,1,0,0,1,1,1,1,0,1,0,0,1,1,0,0,0,1,0,0,0,1,1,1,1,1,1,0,1,0,0,0,1,1,1,1,0,1,0,1,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,1,0,1,0,0,1,0,1,0,0,0,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,0,0,1,0,0,1,1,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,0,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,0,1,0,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,0,1,1,1,1,1,1,0,0,1,1,0,0,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,0,0,1,1,1,1,0,1,1,1,0,1,1,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,1,1,1,0,0,1,0,0,1,0,1,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,1,0,0,1,0"
        # str_masks = mask.split(';')
        # mask = "0,1,1,0,0,1,1,0;0,1,0,1,0,1,1,1,1,0,0,1,0,0;0,0,1,1,1,0,0,0,1,1" # 10K
        str_masks = mask.mask.split(';')
        masks = list(map(lambda str_mask: self.convert_str_to_mask(str_mask).astype(bool), str_masks))
        return masks
