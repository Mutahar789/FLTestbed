# Database object controllers
# Generic imports
import hashlib
import logging
import uuid
from datetime import datetime
import traceback
from ...core.codes import CYCLE, MSG_FIELD
from ...core.exceptions import ProtocolNotFoundError
from ..cycles import cycle_manager
from ..models import model_manager
from ..processes import process_manager
from ..workers import worker_manager
from ..activations import activation_manager
from ..metrics import metrics_manager
from ..fcm import fcm_push_manager
from ..fcm.fcm_push_manager import PUSH_TYPES
import numpy as np
import torch as th
class FLController:
    """This class implements controller design pattern over the federated
    learning processes."""

    def __init__(self):
        pass

    def create_process(
        self,
        model,
        client_plans,
        client_config,
        server_config,
        server_averaging_plan,
        client_protocols=None,
    ):
        """ Register a new federated learning process
            Args:
                model: Model object.
                client_plans : an object containing syft plans.
                client_protocols : an object containing syft protocols.
                client_config: the client configurations
                server_averaging_plan: a function that will instruct PyGrid on how to average model diffs that are returned from the workers.
                server_config: the server configurations
            Returns:
                process : FLProcess Instance.
            Raises:
                FLProcessConflict (PyGridError) : If Process Name/Version already exists.
        """
        cycle_len = server_config["cycle_length"]

        # Create a new federated learning process
        # 1 - Create a new process
        # 2 - Save client plans/protocols
        # 3 - Save Server AVG plan
        # 4 - Save Client/Server configs
        _process = process_manager.create(
            client_config,
            client_plans,
            client_protocols,
            server_config,
            server_averaging_plan,
        )

        unserialize = model_manager.unserialize_model_params(model)
        prune_percentage = server_config.get("prune_percentage", 0.5)
        seed = server_config.get("seed", 1549786796)

        # mask = model_manager.create_mask(unserialize[0], prune_percentage, seed)
        logging.info("fl_controller logging mask")
        mask = model_manager.create_mask(unserialize, seed, prune_percentage) ## for CNN

        logging.info(f"{mask} for 6.6M")

        # Save Model
        # Define the initial version (first checkpoint)
        _model = model_manager.create(model, _process, mask)

        # Create the initial cycle
        _cycle = cycle_manager.create(_process.id, _process.version, cycle_len)

        return _process

    def last_cycle(self, worker_id: str, name: str, version: str) -> int:
        """Retrieve the last time the worker participated from this cycle.

        Args:
            worker_id: Worker's ID.
            name: Federated Learning Process Name.
            version: Model's version.
        Return:
            last_participation: Index of the last cycle assigned to this worker.
        """
        process = process_manager.first(name=name, version=version)
        return cycle_manager.last_participation(process, worker_id)

    def cycle_already_assigned(self, _fl_process, worker_id):
        # Retrieve the last cycle used by this fl process/ version
        _cycle = cycle_manager.last(_fl_process.id, None)

        # Check if already exists a relation between the worker and the cycle.
        _assigned = cycle_manager.is_assigned(worker_id, _cycle.id)
        return _assigned

    def assign(self, name: str, version: str, worker, last_participation: int, cycle_start_request_key):
        """ Assign a new worker the specified federated training worker cycle
            Args:
                name: Federated learning process name.
                version: Federated learning process version.
                worker: Worker Object.
                last_participation: The last time that this worker worked on this fl process.
            Return:
                last_participation: Index of the last cycle assigned to this worker.
        """
        _accepted = False

        if version:
            _fl_process = process_manager.first(name=name, version=version)
        else:
            _fl_process = process_manager.last(name=name)

        server_config, client_config = process_manager.get_configs(
            name=name, version=version
        )

        # Retrieve the last cycle used by this fl process/ version
        _cycle = cycle_manager.last(_fl_process.id, None)

        # Check if already exists a relation between the worker and the cycle.
        _assigned = cycle_manager.is_assigned(worker.id, _cycle.id)
        logging.info(
            f"Worker {worker.id} is already assigned to cycle {_cycle.id}: {_assigned}"
        )

        # Check bandwidth
        _comp_bandwidth = worker_manager.is_eligible(worker.id, server_config)

        # Check if the current worker is allowed to join into this cycle

        _allowed = True#cycle_start_request_key == _cycle.cycle_start_key if cycle_start_request_key is not None else False

        # TODO wire intelligence
        # (
        #     last_participation + server.config["do_not_reuse_workers_until_cycle"]
        #     >= _cycle.sequence
        # )

        # _allowed = True

        _accepted = (not _assigned) and _comp_bandwidth and _allowed
        logging.info(f"Worker is _assigned: {_assigned}, _allowed {_allowed}")
        logging.info(f"Worker is accepted: {_accepted}")

        # _accepted = True
        if _accepted:
            # Assign
            # 1 - Generate new request key

            # 2 - Assign the worker with the cycle.
            key = self._generate_hash_key(uuid.uuid4().hex)
            _worker_cycle = cycle_manager.assign(worker, _cycle, key)

            # Create a plan dictionary
            _plans = process_manager.get_plans(
                fl_process_id=_fl_process.id, is_avg_plan=False
            )

            print("=================================")
            print("=================================")
            print("=================================")
            is_force_pruning = server_config.get("is_force_pruning", False)
            if worker.is_slow or is_force_pruning:
                    if "training_plan_small" in _plans.keys():
                        plan_id = _plans["training_plan_small"]
                        del _plans["training_plan"]
                        del _plans["training_plan_small"]
                        _plans["training_plan"] = plan_id

            # logging.info(f"_plans.keys() ==================== {_plans}")
            # exit()

            print("=================================")
            print("=================================")
            print("=================================")

            # Create a protocol dictionary
            try:
                _protocols = process_manager.get_protocols(fl_process_id=_fl_process.id)
            except ProtocolNotFoundError:
                # Protocols are optional
                _protocols = {}

            # Get model ID
            _model = model_manager.get(fl_process_id=_fl_process.id)
            return {
                CYCLE.STATUS: "accepted",
                CYCLE.KEY: _worker_cycle.request_key,
                CYCLE.CYCLE_ID: _worker_cycle.cycle_id,
                CYCLE.VERSION: _cycle.version,
                MSG_FIELD.MODEL: name,
                CYCLE.PLANS: _plans,
                CYCLE.PROTOCOLS: _protocols,
                CYCLE.CLIENT_CONFIG: client_config,
                MSG_FIELD.MODEL_ID: _model.id,
            }
        else:
            n_completed_cycles = cycle_manager.count(
                fl_process_id=_fl_process.id, is_completed=True
            )

            _max_cycles = server_config["num_cycles"]

            response = {CYCLE.STATUS: "rejected"}

            # If it's not the last cycle, add the remaining time to the next cycle.
            if n_completed_cycles < _max_cycles:
                remaining = _cycle.end - datetime.now()
                response[CYCLE.TIMEOUT] = str(remaining)

            return response

    def _generate_hash_key(self, primary_key: str) -> str:
        """Generate SHA256 Hash to give access to the cycle.

        Args:
            primary_key : Used to generate hash code.
        Returns:
            hash_code : Hash in string format.
        """
        return hashlib.sha256(primary_key.encode()).hexdigest()

    def submit_diff(self, worker_id: str, request_key: str, diff: bin, num_samples: int, worker):
        """Submit worker model diff to the assigned cycle.

        Args:
            worker_id: Worker's ID.
            request_key: request (token) used by this worker during this cycle.
            diff: Model params trained by this worker.
        Raises:
            ProcessLookupError : If Not found any relation between the worker/cycle.
        """

        return cycle_manager.submit_worker_diff(worker_id, request_key, diff, num_samples, worker)

    def save_activations(self,fl_process_id: int, activations: bin, worker: str, cycles_for_avg: int):
        """
            Whenever client report the updated training parameters on server, client also sends activations for intellegent model pruning.
            Activations are being sent until boot strap rounds are completed.
        """

        activation = None
        unserialized_activation = model_manager.unserialize_model_params(activations)
        # logging.info(f"======================== unserialized_activation  {unserialized_activation}")
        try:
            if worker.is_slow:
                activation_with_original_params = model_manager.convert_pruned_activations_to_original(unserialized_activation[0])
                activation = activation_with_original_params
            else:
                activation = activations

            # logging.info(f"======================== after conversion to serialzed buffer: reconstruct  {model_manager.unserialize_model_params(activation)}")


            activation_manager.save_activations(fl_process_id, activation, worker.id, cycles_for_avg, worker.is_slow)
        except Exception as e:
            logging.info(f"============================ exception {str(e)} {str(e) + traceback.format_exc()}")
            raise SystemExit(0)

    def get_activations(self):
        # fl_process_id: int, activations: bin, worker_id: str, cycles_for_avg: int
        activations = activation_manager.get_activations()

    def save_metric(self, cycle_id, worker_id, test_accuracy_list, model_size, avg_epoch_time, total_train_time, is_slow, trained_epochs, model_download_time, model_report_time, model_download_size, model_report_size):
        metrics_manager.save_metric(cycle_id, worker_id, test_accuracy_list, model_size, avg_epoch_time, total_train_time, is_slow, trained_epochs, model_download_time, model_report_time, model_download_size, model_report_size)

        # if interrupt_training:
        #     fl_process = process_manager.get(id=1)
        #     _cycle = cycle_manager.last(fl_process[0].id, None)
        #     fcm_push_manager.trigger_event(fl_process[0].name, fl_process[0].version, PUSH_TYPES.TYPE_CYCLE_COMPLETED, _cycle, cycle_manager, fl_process)

    def _average_plan_diffs(self):
        model_name = "mnist"
        model_version = "1.0"
        _fl_process = process_manager.first(name=model_name, version=model_version)
        # Retrieve the last cycle used by this fl process/ version
        _cycle = cycle_manager.last(_fl_process.id, None)

        kwargs = {"name": model_name}
        if model_version is not None:
            kwargs["version"] = model_version

        server_config, _ = process_manager.get_configs(**kwargs)

        # cycle_manager._average_plan_diffs(server_config, _cycle)

        _cycle = cycle_manager.last(_fl_process.id, None)
        try:
            metrics_received = metrics_manager.query(cycle_id=_cycle.id)
        except Exception as e:
            print("Exception: ", str(e))
            exit()

        high_metric_count = 0
        low_metric_count = 0
        for metric in metrics_received:
            if not metric.is_slow:
                high_metric_count += 1
            else:
                low_metric_count += 1

        metrics_count = high_metric_count + low_metric_count
        # if metrics_count == 4:
        #     cycle_manager._average_plan_diffs(server_config, _cycle)

        metrics_count = high_metric_count + low_metric_count
        logging.info(f"================================ metrics_count: {metrics_count}")

        fl_process = process_manager.get(id=_cycle.fl_process_id)


        if metrics_count == 2:
            cycle_manager._average_plan_diffs(server_config, _cycle)
        elif high_metric_count == 1:
            fcm_push_manager.trigger_event(fl_process[0].name, fl_process[0].version, PUSH_TYPES.TYPE_INTERRUPT_CYCLE_COMPLETED, _cycle, self,fl_process)

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
        cycle_manager._average_plan_diffs(server_config, _cycle, True)
