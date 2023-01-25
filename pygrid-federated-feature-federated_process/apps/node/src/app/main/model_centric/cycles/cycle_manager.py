# Cycle module imports
import json
import logging

# Generic imports
from datetime import datetime, timedelta
from functools import reduce

import torch as th
from sqlalchemy import and_
from ...core.exceptions import CycleNotFoundError

# PyGrid modules
from ...core.warehouse import Warehouse
from ..models import model_manager
from ..processes import process_manager

from ..syft_assets import PlanManager
from ..tasks.cycle import complete_cycle, run_task_once
from .cycle import Cycle
from .worker_cycle import WorkerCycle

# trigger events module
from ..tasks.cycle import complete_cycle, run_task_once
from ..fcm import fcm_push_manager
from ..fcm.fcm_push_manager import PUSH_TYPES
from ..activations import activation_manager
from ..metrics import metrics_manager
from ..metrics.metrics import  Metrics
import numpy as np

th.set_printoptions(precision=8)

class CycleManager:
    def __init__(self):
        self._cycles = Warehouse(Cycle)
        self._worker_cycles = Warehouse(WorkerCycle)
        self._metrics = Warehouse(Metrics)

    def create(self, fl_process_id: int, version: str, cycle_time: int):
        """Create a new federated learning cycle.

        Args:
            fl_process_id: FL Process's ID.
            version: Version (?)
            cycle_time: Remaining time to finish this cycle.
        Returns:
            fd_cycle: Cycle Instance.
        """
        _new_cycle = None

        # Retrieve a list of cycles using the same model_id/version
        sequence_number = len(
            self._cycles.query(fl_process_id=fl_process_id, version=version)
        )
        _now = datetime.now()
        _end = _now + timedelta(seconds=cycle_time) if cycle_time is not None else None
        _new_cycle = self._cycles.register(
            start=_now,
            end=_end,
            sequence=sequence_number + 1,
            version=version,
            fl_process_id=fl_process_id,
        )

        return _new_cycle

    def last_participation(self, process: int, worker_id: str):
        """Retrieve the last time the worker participated from this cycle.

        Args:
            process: Federated Learning Process.
            worker_id: Worker's ID.
        Returns:
            last_participation: last cycle.
        """
        _cycles = self._cycles.query(fl_process_id=process.id)

        last = 0
        if not len(_cycles):
            return last

        for cycle in _cycles:
            worker_cycle = self._worker_cycles.first(
                cycle_id=cycle.id, worker_id=worker_id
            )
            if worker_cycle and cycle.sequence > last:
                last = cycle.sequence

        return last

    def last(self, fl_process_id: int, version: str = None):
        """Retrieve the last not completed registered cycle.

        Args:
            fl_process_id: Federated Learning Process ID.
            version: Model's version.
        Returns:
            cycle: Cycle Instance / None
        """
        if version:
            _cycle = self._cycles.last(
                fl_process_id=fl_process_id, version=version, is_completed=False
            )
        else:
            _cycle = self._cycles.last(fl_process_id=fl_process_id, is_completed=False)


        if not _cycle:
            raise CycleNotFoundError

        return _cycle

    def delete(self, **kwargs):
        """Delete a registered Cycle.

        Args:
            model_id: Model's ID.
        """
        self._cycles.delete(**kwargs)

    def is_assigned(self, worker_id: str, cycle_id: int):
        """Check if a workers is already assigned to an specific cycle.

        Args:
            worker_id : Worker's ID.
            cycle_id : Cycle's ID.
        Returns:
            result : Boolean Flag.
        """
        return self._worker_cycles.first(worker_id=worker_id, cycle_id=cycle_id) != None

    def assign(self, worker, cycle, hash_key: str):
        _worker_cycle = self._worker_cycles.register(
            worker=worker, cycle=cycle, request_key=hash_key
        )

        return _worker_cycle

    def validate(self, worker_id: str, cycle_id: int, request_key: str):
        """Validate Worker's request key.

        Args:
            worker_id: Worker's ID.
            cycle_id: Cycle's ID.
            request_key: Worker's request key.
        Returns:
            result: Boolean flag
        Raises:
            CycleNotFoundError (PyGridError) : If not found any relation between the worker and cycle.
        """
        _worker_cycle = self._worker_cycles.first(
            worker_id=worker_id, cycle_id=cycle_id
        )

        if not _worker_cycle:
            raise CycleNotFoundError

        return _worker_cycle.request_key == request_key

    def count(self, **kwargs):
        return self._cycles.count(**kwargs)

    def count_worker_cycles(self, cycle_id):
        return self._worker_cycles.count(cycle_id=cycle_id)

    def submit_worker_diff(self, worker_id: str, request_key: str, diff: bin, num_samples: int, worker):
        """Submit reported diff
           Args:
                worker_id: Worker's ID.
                request_key: request (token) used by this worker during this cycle.
                diff: Model params trained by this worker.
           Returns:
                cycle_id : Cycle's ID.
           Raises:
                ProcessLookupError : If Not found any relation between the worker/cycle.
        """
        _worker_cycle = self._worker_cycles.first(
            worker_id=worker_id, request_key=request_key
        )

        if not _worker_cycle:
            raise ProcessLookupError

        cycle = self._cycles.first(id=_worker_cycle.cycle_id)
        server_config, _ = process_manager.get_configs(id=cycle.fl_process_id)
        is_pruning_enabled = server_config.get("is_pruning_enabled", True)
        is_force_pruning = server_config.get("is_force_pruning", False)

        if (worker.is_slow and is_pruning_enabled) or is_force_pruning:
            unserialized_diff = model_manager.unserialize_model_params(diff)
            # list_model_params = []
            # for param in unserialized_diff:
            #     list_model_params.append(param.data.numpy())
            #
            # np_weights = np.array(list_model_params)
            # np.save(f'before_convert_weights_to_org', np_weights)
            #
            # logging.info(f"unserialized_diff", unserialized_diff)
            diff_with_original_params = model_manager.convert_pruned_model_to_original(unserialized_diff)

            _worker_cycle.diff = diff_with_original_params
        else:
            _worker_cycle.diff = diff

        _worker_cycle.is_completed = True
        _worker_cycle.completed_at = datetime.utcnow()
        _worker_cycle.num_samples = num_samples
        _worker_cycle.is_slow = worker.is_slow
        self._worker_cycles.update()

        # Run cycle end task async to we don't block report request
        # (for prod we probably should be replace this with Redis queue + separate worker)
        run_task_once("complete_cycle", complete_cycle, self, _worker_cycle.cycle_id)



    def complete_cycle(self, cycle_id: int):
        """Checks if the cycle is completed and runs plan avg."""
        # logging.info("running complete_cycle for cycle_id: %s" % cycle_id)

        cycle = self._cycles.first(id=cycle_id)
        server_config, _ = process_manager.get_configs(id=cycle.fl_process_id)
        optimizer = server_config.get("optimizer", "hasaas")

        if optimizer == "fedavg" or optimizer == "fedprox":
            self.interrupt_training()
        else:
           # logging.info("found cycle: %s" % str(cycle))

            if cycle.is_completed:
                logging.info("cycle is already completed!")
                event_type = PUSH_TYPES.TYPE_CYCLE_COMPLETED
                _fl_process = process_manager.first(name="mnist", version="1.0")
                _cycle = self.last(_fl_process.id, None)
                fcm_push_manager.trigger_event("mnist", "1.0", event_type, _cycle, self, _fl_process)
                return


            # logging.info("server_config: %s" % json.dumps(server_config, indent=2))

            received_diffs = self._worker_cycles.count(cycle_id=cycle_id, is_completed=True)
            # logging.info("# of diffs: %d" % received_diffs)

            min_diffs = server_config.get("min_diffs", None)
            max_diffs = server_config.get("max_diffs", None)

            hit_diffs_limit = (
                received_diffs >= max_diffs if max_diffs is not None else False
            )
            hit_time_limit = datetime.now() >= cycle.end if cycle.end is not None else False
            no_limits = max_diffs is None and cycle.end is None
            has_enough_diffs = (
                received_diffs >= min_diffs if min_diffs is not None else True
            )

            ready_to_average = has_enough_diffs and (
                no_limits or hit_diffs_limit or hit_time_limit
            )

            no_protocol = True  # only deal with plans for now

            # logging.info("cycle: %s" % str(cycle))
            #
            # logging.info("hit_diffs_limit: %d" % int(hit_diffs_limit))
            # logging.info("no_limits: %d" % int(no_limits))
            # logging.info("hit_time_limit: %d" % int(hit_time_limit))
            #
            # logging.info("ready_to_average: %d" % int(ready_to_average))

            if ready_to_average and no_protocol:
                self._average_plan_diffs(server_config, cycle)

    def interrupt_training(self):
        model_name = "mnist"
        model_version = "1.0"
        _fl_process = process_manager.first(name=model_name, version=model_version)
        # Retrieve the last cycle used by this fl process/ version
        _cycle = self.last(_fl_process.id, None)

        kwargs = {"name": model_name}
        if model_version is not None:
            kwargs["version"] = model_version

        server_config, _ = process_manager.get_configs(**kwargs)
        # self._average_plan_diffs(server_config, _cycle, True)

        try:
            metrics_received = self._metrics.query(cycle_id=_cycle.id)
        except Exception as e:
            print("Exception: ", str(e))
            exit()
        fl_process = process_manager.get(id=_cycle.fl_process_id)
        event_type = PUSH_TYPES.TYPE_INTERRUPT_CYCLE_COMPLETED


        high_metric_count = 0
        low_metric_count = 0
        for metric in metrics_received:
            if not metric.is_slow:
                high_metric_count += 1
            else:
                low_metric_count += 1

        metrics_count = high_metric_count + low_metric_count
        logging.info(f"================================ metrics_count: {metrics_count}")

        # self._average_plan_diffs(server_config, _cycle)
        if metrics_count == 2:
            self._average_plan_diffs(server_config, _cycle)
        # elif high_metric_count == 1:
        #     fcm_push_manager.trigger_event(fl_process[0].name, fl_process[0].version, event_type, _cycle, self,fl_process)

    def update_model(self, raw_diffs):
        total_weight = 0
        all_clients_diff_one_layer = raw_diffs[0]
        for diff, weight in all_clients_diff_one_layer:
            total_weight += weight
        base = [0] * len(raw_diffs)
        for layer_no, diff in enumerate(raw_diffs):
            for (client_model, client_samples) in diff:
                base[layer_no] += (client_samples * client_model)
        averaged_soln = [v / total_weight for v in base]
        return averaged_soln

    def where(self, filterArr, arr1, arr2, axis=0):
        if arr1.shape != arr2.shape:
            print('Error: arr1 and arr2 must have equal shapes')
            return
        res = th.zeros(arr1.shape, dtype=arr1.dtype)
        for i in range(filterArr.shape[0]):
            if axis == 0:
                res[i] = arr1[i] if filterArr[i] else arr2[i]
            elif axis == 1:
                res[:, i] = arr1[:, i] if filterArr[i] else arr2[:, i]
        return res

    def aggregate_conv_dense_layer(self, weights_all, weights_high, layer_no, mask):
        weights_all[layer_no] = self.where(mask, weights_all[layer_no], weights_high[layer_no])
        weights_all[layer_no + 1] = self.where(mask, weights_all[layer_no + 1], weights_high[layer_no + 1])
        weights_all[layer_no + 2] = self.where(mask, weights_all[layer_no + 2], weights_high[layer_no + 2], axis=1)
        return weights_all

    def get_averaged_params(self, updates, weights, totalWeight):
        aggregatedParams = [th.zeros(param.shape) for param in updates[0]]
        for i, params in enumerate(updates):
            for layer_index, param in enumerate(params):
                aggregatedParams[layer_index] += ((param * weights[i]) / totalWeight)

        return aggregatedParams

    def weighted_std(self, average, values, weights, totalWeight):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        std = [th.zeros(param.shape) for param in average]
        for i, value in enumerate(values):
            totalWeight += weights[i]

            for layer_no, layer in enumerate(value):
                variance_layer = (layer - average[layer_no]) ** 2
                variance_layer = (variance_layer * weights[i]) / totalWeight
                std[layer_no] += th.sqrt(variance_layer)

        return std

    def get_aggregated_weights(self, masks, average_params, average_params_high):
        for i, mask in enumerate(masks):
            if i == 0 or i == 2:
                averaged_soln = self.aggregate_conv_dense_layer(average_params, average_params_high, i * 2, mask)
            elif i == 1:
                numFilters = average_params[2].shape[0]
                numNeuronsDense = average_params[4].shape[0]

                average_params[4] = average_params[4].reshape(-1, numFilters, 7, 7)
                average_params_high[4] = average_params_high[4].reshape(-1, numFilters, 7, 7)

                averaged_soln = self.aggregate_conv_dense_layer(average_params, average_params_high, i * 2, mask)

                average_params[4] = average_params[4].reshape(numNeuronsDense, -1)
                average_params_high[4] = average_params_high[4].reshape(numNeuronsDense, -1)

        return averaged_soln

    def get_aggregated_weights_clt(self, masks, average_params, average_params_high, std_params, std_params_high):
        for i, mask in enumerate(masks):
            if i == 0 or i == 2:
                averaged_soln = self.aggregate_conv_dense_layer(average_params, average_params_high, i * 2, mask)
                std_soln = self.aggregate_conv_dense_layer(std_params, std_params_high, i * 2, mask)
            elif i == 1:
                numFilters = average_params[2].shape[0]
                numNeuronsDense = average_params[4].shape[0]

                average_params[4] = average_params[4].reshape(-1, numFilters, 7, 7)
                average_params_high[4] = average_params_high[4].reshape(-1, numFilters, 7, 7)

                std_params[4] = std_params[4].reshape(-1, numFilters, 7, 7)
                std_params_high[4] = std_params_high[4].reshape(-1, numFilters, 7, 7)

                averaged_soln = self.aggregate_conv_dense_layer(average_params, average_params_high, i * 2, mask)
                std_soln = self.aggregate_conv_dense_layer(std_params, std_params_high, i * 2, mask)

                average_params[4] = average_params[4].reshape(numNeuronsDense, -1)
                average_params_high[4] = average_params_high[4].reshape(numNeuronsDense, -1)

                std_params[4] = std_params[4].reshape(numNeuronsDense, -1)
                std_params_high[4] = std_params_high[4].reshape(numNeuronsDense, -1)

        return averaged_soln, std_soln

    def _average_plan_diffs(self, server_config: dict, cycle, interrupt_required = False):
        """skeleton code Plan only.

        - get cycle
        - track how many has reported successfully
        - get diffs: list of (worker_id, diff_from_this_worker) on cycle._diffs
        - check if we have enough diffs? vs. max_worker
        - if enough diffs => average every param (by turning tensors into python matrices => reduce th.add => torch.div by number of diffs)
        - save as new model value => M_prime (save params new values)
        - create new cycle & new checkpoint
        at this point new workers can join because a cycle for a model exists
        """

        reports_to_average = self._worker_cycles.query(
            cycle_id=cycle.id, is_completed=True
        )

        if reports_to_average is None or len(reports_to_average) == 0:
            return

        logging.info("start diffs averaging!")
        logging.info("cycle: %s" % str(cycle))
        logging.info("fl id: %d" % cycle.fl_process_id)
        _model = model_manager.get(fl_process_id=cycle.fl_process_id)
        # logging.info("model: %s" % str(_model))
        model_id = _model.id
        # logging.info("model id: %d" % model_id)
        _checkpoint = model_manager.load(model_id=model_id)
        # logging.info("current checkpoint: %s" % str(_checkpoint))
        model_params = model_manager.unserialize_model_params(_checkpoint.value)
        # logging.info("model params shapes: %s" % str([p.shape for p in model_params]))



        # list_model_params = []
        # for param in model_params:
        #     list_model_params.append(param.data.numpy())
        #
        # testbed_pruned_model = prune_model(list_model_params)
        # testbed_orig_model = convert_pruned_model_to_original(testbed_pruned_model)
        #
        # print(testbed_pruned_model)
        #
        # diffs = []  # np.array()
        #
        # diffs.append((500, np.array(testbed_orig_model), 1))
        # diffs.append((500, np.array(list_model_params), 0))

        # logging.info(f"model_params {model_params}")
        logging.info(
            "model_params shapes: %s"
            % str([p.shape for p in model_params])
        )

        optimizer = server_config.get("optimizer", "hasaas")


        avg_plan_rec = False
        if avg_plan_rec and avg_plan_rec.value:
            diffs = [
                model_manager.unserialize_model_params(report.diff)
                for report in reports_to_average
            ]
            logging.info("Doing hosted avg plan")
            avg_plan = PlanManager.deserialize_plan(avg_plan_rec.value)

            # check if the uploaded avg plan is iterative or not
            iterative_plan = server_config.get("iterative_plan", False)

            # diffs if list [diff1, diff2, ...] of len == received_diffs
            # each diff is list [param1, param2, ...] of len == model params
            # diff_avg is list [param1_avg, param2_avg, ...] of len == model params
            iterative_plan = True
            if iterative_plan:
                diff_avg = diffs[0]
                for i, diff in enumerate(diffs[1:]):
                    diff_avg = avg_plan(list(diff_avg), diff, th.tensor([i + 1]))
            else:

                diff_avg = avg_plan(diffs)

                # apply avg diff!

            _updated_model_params = [
                model_param - diff_param
                for model_param, diff_param in zip(model_params, diff_avg)
            ]
        else:
            # Fallback to simple hardcoded avg plan
            logging.info("Doing hardcoded avg plan")

            #
            # # raw_diffs is [param1_diffs, param2_diffs, ...] with len == num of model params
            # # each param1_diffs is [ diff1, diff2, ... ] with len == num of received_diffs
            # # diff_avg going to be [ param1_diffs_avg, param2_diffs_avg, ...] with len =model_params= num of model params
            # # where param1_diffs_avg is avg of tensors in param1_diffs
            # # sums = add every column entry with other tensor same column entry in the end it will give number of output channels x input features.
            # ####
            # # LEAF FedAvg Formula
            # ####
            # # logging.info("raw diffs lengths: %s" % str([len(row) for row in raw_diffs]))
            # #
            total_weight = 0
            # raw_diffs = [
            #     [(diff[0][layer_no], diff[1]) for diff in diffs]
            #     for layer_no in range(len(model_params))
            # ]

            # logging.info(f"=====================>>>>>>>>>>>>>>>>>>>>>> raw_diffs {raw_diffs}")


            # all_clients_diff_one_layer = raw_diffs[0]
            # for diff, weight in all_clients_diff_one_layer:
            #     total_weight += weight
            #
            # # logging.info(f"==================>>>>>>>> total_weight {total_weight}, len(raw_diffs) {len(raw_diffs)} len(raw_diffs[0]) {len(raw_diffs[0])}")
            # # 3 x 60 x 10 = for every client if clients are 10 = 10 arrays
            #
            # # total_weight = 2400
            # weighted_diffs = [[model * samples for (model, samples) in diff] for diff in raw_diffs]
            #
            #
            #
            # # logging.info(f"==================>>>>>>>>>>>>> weighted_diffs {weighted_diffs}")
            #
            # # 3 x 60
            # weighted_sums = [reduce(th.add, diff) for diff in weighted_diffs]
            #
            # # logging.info(f"==================>>>>>>>>>>>>> weighted_sums {weighted_sums}")
            #
            # diff_avg = [th.div(param, total_weight) for param in weighted_sums]
            #
            # # logging.info(f"==================>>>>>>>>>>>>> diff_avg {diff_avg}")
            #
            # # exit()



            if optimizer == "fedavg" or optimizer == "fedprox":
                diffs = [(model_manager.unserialize_model_params(report.diff), report.num_samples) for report in reports_to_average]
                raw_diffs = [ [(diff[0][layer_no], diff[1]) for diff in diffs] for layer_no in range(len(model_params)) ]
                _updated_model_params = self.update_model(raw_diffs)
            else:
                diffs = [(report.num_samples, model_manager.unserialize_model_params(report.diff), report.is_slow) for report in reports_to_average]
                str_masks = model_manager.get_mask().mask
                masks = list(map(lambda str_mask: model_manager.convert_str_to_mask(str_mask), str_masks.split(';')))

                allNumSamples = []
                allHighNumSamples = []

                allModelParams = []
                highModelParams = []

                all_types = []

                totalNumSamples = 0
                totalHighNumSamples = 0

                for item in diffs:
                    allNumSamples.append(item[0])
                    allModelParams.append(item[1])
                    all_types.append(item[2])
                    totalNumSamples += item[0]
                    if item[2] == 0:
                        highModelParams.append(item[1])
                        allHighNumSamples.append(item[0])
                        totalHighNumSamples += item[0]

                averaged_params = self.get_averaged_params(allModelParams, allNumSamples, totalNumSamples)
                averaged_params_high = self.get_averaged_params(highModelParams, allHighNumSamples, totalHighNumSamples)
                std_params = self.weighted_std(averaged_params, allModelParams, allNumSamples, totalNumSamples)
                std_params_high = self.weighted_std(averaged_params_high, highModelParams, allHighNumSamples, totalHighNumSamples)

                if(cycle.id == 1):
                    logging.info(f">>>>>>>>>>>>>>>>>>>>>>> len(diffs) {len(diffs)}")

                aggregatedParams, aggregatedStd = self.get_aggregated_weights_clt(masks, averaged_params, averaged_params_high, std_params, std_params_high)
                new_params = [th.normal(mean=aggregatedParams[i], std=th.div(aggregatedStd[i], np.sqrt(len(diffs)))) for i in range(len(aggregatedParams))]
                _updated_model_params = new_params

        # logging.info(f"_updated_model_params {_updated_model_params}")
        logging.info("_updated_model_params shapes: %s"% str([p.shape for p in _updated_model_params]))

        # make new checkpoint
        serialized_params = model_manager.serialize_model_params(_updated_model_params)
        _new_checkpoint = model_manager.save(model_id, serialized_params)
        # logging.info("new checkpoint: %s" % str(_new_checkpoint))

        # mark current cycle completed
        cycle.is_completed = True
        cycle.completed_at = datetime.now()
        self._cycles.update()

        completed_cycles_num = self._cycles.count(
            fl_process_id=cycle.fl_process_id, is_completed=True
        )
        # logging.info("completed_cycles_num: %d" % completed_cycles_num)
        max_cycles = server_config.get("num_cycles", 0)

        server_config, _ = process_manager.get_configs(id=cycle.fl_process_id)

        bootstrap_rounds = int(server_config['bootstrap_rounds'])
        prune_percentage = server_config['prune_percentage']
        is_pruning_enabled = server_config['is_pruning_enabled']

        if bootstrap_rounds == int(cycle.id)  and is_pruning_enabled:
            updated_mask = self.updated_cnn_mask(_updated_model_params)
            self.update_mask(model_id, updated_mask, prune_percentage)

        event_type = None
        new_cycle = None

        if int(cycle.id) + 1 >= max_cycles:
            list_model_params = []
            for param in _updated_model_params:
                list_model_params.append(param.data.numpy())

            np_weights = np.array(list_model_params)
            np.save(f'weights_for_round_{cycle.id + 1}', np_weights)


        if completed_cycles_num < max_cycles or max_cycles == 0:
            # make new cycle
            cycle_length = server_config.get("cycle_length")
            _new_cycle = self.create(
                cycle.fl_process_id, cycle.version, cycle_length
            )
            logging.info("new cycle: %s" % str(_new_cycle))
            if interrupt_required and (optimizer == "fedavg" or optimizer == "fedprox"):
                event_type = PUSH_TYPES.TYPE_INTERRUPT_CYCLE_COMPLETED
            else:
                event_type = PUSH_TYPES.TYPE_CYCLE_COMPLETED
            new_cycle = _new_cycle
        else:
            logging.info("FL is done!")
            event_type = PUSH_TYPES.TYPE_FL_COMPLETED

        fl_process = process_manager.get(id=cycle.fl_process_id)
        fcm_push_manager.trigger_event(fl_process[0].name, fl_process[0].version, event_type, new_cycle, self, fl_process)

    def update(self):
        """Update Cycle Attributes."""
        return self._cycles.update()

    def check_cycle_expiration(self):
        """
        This function checks whether current active cycle is exprired or not. If it is expired then
        it does the following 2 steps:
        1. Resets the current cycle by setting start = datetime.now() and end = start + cycle_length according to configuration
        2. Delete the workers from worker_cycle table by ignoring updates for the current cycle and making those workers available for resetted cycle
        """



        model_name = "mnist"
        model_version = "1.0"
        _fl_process = process_manager.first(name=model_name, version=model_version)
        # Retrieve the last cycle used by this fl process/ version
        _cycle = self.last(_fl_process.id, None)

        kwargs = {"name": model_name}
        if model_version is not None:
            kwargs["version"] = model_version

        server_config, _ = process_manager.get_configs(**kwargs)

        if datetime.now() > _cycle.end:
            logging.info("Cycle expired. Reset the cycle and workers for this cycle.")
            # self.reset_cycle(server_config, _cycle)
            # self.delete_workers_resetted_cycle(_cycle)
            #
            # fcm_push_manager.trigger_event(model_name, model_version, PUSH_TYPES.TYPE_CYCLE_RESET, _cycle, self, _fl_process)

            self._average_plan_diffs(server_config, _cycle, True)

    def reset_cycle(self, server_config, _cycle):
        """
        Reset the cycle start and end time
        """
        _now = datetime.now()
        cycle_len = server_config["cycle_length"]
        _end = _now + timedelta(seconds=cycle_len) if cycle_len is not None else None

        _cycle.cycle_start_key = ''
        _cycle.start = _now
        _cycle.end = _end

        self.update()

    def delete_workers_resetted_cycle(self, _cycle):
        self._worker_cycles.delete_all_where(cycle_id=_cycle.id)

    def updated_cnn_mask(self, _updated_model_params):
        weights = _updated_model_params
        str_masks = model_manager.get_mask().mask
        masks = list(map(lambda str_mask: model_manager.convert_str_to_mask(str_mask), str_masks.split(';')))
        info = {}
        # info['conv1/kernel:0'] = [0, 32, masks[0]]
        # info['conv_last/kernel:0'] = [2, 64, masks[1]]

        info['conv1/kernel:0'] = [0, 16, masks[0]]
        info['conv_last/kernel:0'] = [2, 32, masks[1]]

        for layer in info:
            if "conv" in layer:
                mask = info[layer][2]
                layer_no = info[layer][0]
                total_filters = info[layer][1]
                filter_val = []

                for i in range(total_filters):
                    temp = weights[layer_no][i, :, :, :]
                    norm = np.linalg.norm(temp.flatten(), ord=1)
                    filter_val.append(norm)

                m = np.ones(len(filter_val), dtype=int)
                dropN = total_filters - np.count_nonzero(mask)
                rind = np.array(filter_val).argsort()[:dropN]
                m[rind] = 0
                info[layer][2] = m
        # updated_cnn_mask = f"{masks[0]};{masks[1]};{masks[2]};"
        # model_manager.update_cnn_mask(model_id, updated_cnn_mask)
        # info['conv_last/kernel:0'] = [2, 64, masks[1]]

        masks[0] = info['conv1/kernel:0'][2]
        masks[1] = info['conv_last/kernel:0'][2]
        return masks

    def update_mask(self, model_id, updated_mask_list, prune_percentage):
        """
            Based on bootstrap rounds update the mask for pruning the model
        """

        # activations = activation_manager.get_activations()
        # # logging.info(f"======================= activations {activations}")
        #
        # list = [model_manager.unserialize_model_params(act.Activation.avg_activations)[0] for act in activations]

        all_activations = activation_manager.get_activations()
        # logging.info(f"======================= activations {activations}")

        all_list = [model_manager.unserialize_model_params(act.Activation.avg_activations)[0] for act in all_activations]

        # logging.info(f"========================= list {list}")

        # addition = th.stack(list, dim=0).sum(dim=0)
        # avg_activations = th.div(addition, len(list))

        all_addition = th.stack(all_list, dim=0).sum(dim=0)
        all_avg_activations = th.div(all_addition, len(all_list))

        high_list = []
        for act in all_activations:
            if not act.Worker.is_slow:
                high_list.append(model_manager.unserialize_model_params(act.Activation.avg_activations)[0])

        high_addition = th.stack(high_list, dim=0).sum(dim=0)
        high_avg_activations = th.div(high_addition, len(high_list))

        indices = updated_mask_list[2]

        # avg_activations = [all_avg_activations[i] if int(indices[i]) == 1 else high_avg_activations[i] for i in range(len(indices))]

        avg_activations = th.zeros(high_avg_activations.shape, dtype=high_avg_activations.dtype)
        for i in range(len(indices)):
            if indices[i]:
                avg_activations[i] = all_avg_activations[i]
            else:
                avg_activations[i] = high_avg_activations[i]

        m = np.ones(len(avg_activations), dtype=int)
        N = int(len(avg_activations) * prune_percentage)
        argSortArray = avg_activations.argsort()

        rind = argSortArray[:N].numpy()

        m[rind] = 0

        str_mask = ','.join([str(int(num)) for num in m])

        logging.info(f"=============== genearting mask")

        # str_masks = model_manager.get_mask().mask
        # masks = list(map(lambda str_mask: model_manager.convert_str_to_mask(str_mask), str_masks.split(';')))
        # updated_mask_list[2] = str_mask
        cnn1 = str(np.array2string(updated_mask_list[0],separator=",", formatter={'str_kind':lambda x: "%s" % x})).lstrip('[').rstrip(']')
        cnn2 = str(np.array2string(updated_mask_list[1], separator=",", formatter={'str_kind':lambda x: "%s" % x})).lstrip('[').rstrip(']')
        updated_mask = f"{cnn1};{cnn2};{str_mask}".replace(' ', '')
        model_manager.update_cnn_mask(model_id, updated_mask)

        # logging.info(f"==============================   avg_activations {avg_activations}")
        #
        # raise SystemExit(0)