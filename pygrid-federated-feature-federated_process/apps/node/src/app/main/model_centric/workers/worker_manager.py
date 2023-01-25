# workers module imports
import logging

from ...core.exceptions import WorkerNotFoundError

# PyGrid imports
from ...core.warehouse import Warehouse
from .worker import Worker
from ..processes import process_manager
from ..cycles.worker_cycle import WorkerCycle
from sqlalchemy.sql import not_
from sqlalchemy import and_
from sqlalchemy import or_
from sqlalchemy import case
from ..cycles import cycle_manager

class WorkerManager:
    def __init__(self):
        self._workers = Warehouse(Worker)

    def create(self, worker_id: str):
        """ Register a new worker
            Args:
                worker_id: id used to identify the new worker.
            Returns:
                worker: a Worker instance.
        """
        new_worker = self._workers.register(id=worker_id)
        return new_worker

    def delete(self, **kwargs):
        """Remove a registered worker.

        Args:
            worker_id: Id used identify the desired worker.
        """
        self._workers.delete(**kwargs)

    def get(self, **kwargs):
        """Retrieve the desired worker.

        Args:
            worker_id: Id used to identify the desired worker.
        Returns:
            worker: worker Instance or None if it wasn't found.
        """
        _worker = self._workers.first(**kwargs)

        if not _worker:
            raise WorkerNotFoundError

        return self._workers.first(**kwargs)

    def update(self, worker):
        """Update Workers Attributes."""
        return self._workers.update()

    def is_eligible(self, worker_id: str, server_config: dict):
        """Check if Worker is eligible to join in an new cycle by using its
        bandwidth statistics.

        Args:
            worker_id : Worker's ID.
            server_config : FL Process Server Config.
        Returns:
            result: Boolean flag.
        """
        _worker = self._workers.first(id=worker_id)
        logging.info(
            f"Checking worker [{_worker}] against server_config [{server_config}]"
        )

        # Check bandwidth
        _comp_bandwidth = (
            "minimum_upload_speed" not in server_config
            or _worker.avg_upload >= server_config["minimum_upload_speed"]
        ) and (
            "minimum_download_speed" not in server_config
            or _worker.avg_download >= server_config["minimum_download_speed"]
        )

        logging.info(f"Result of bandwidth check: {_comp_bandwidth}")
        return _comp_bandwidth

    def get_all_workers(self):
        return self._workers.get_all(and_(Worker.fcm_device_token != None,Worker.fcm_device_token.isnot('')))

    def get_all_workers_in_any_cycle(self):
        """
        Get all those workers that are either not in any cycle or were in the cycle and that cycle is completed
        """
        # case([(discriminator == "a", "a")], else_="b")
        # workers = self._workers.get_all_with_join(
        #     Worker, 
        #     WorkerCycle, 
        #     Worker.id == WorkerCycle.worker_id, 
        #     case(
        #         [(and_(WorkerCycle is not None, WorkerCycle.is_completed == True), True),
        #         (and_(WorkerCycle is None, and_(Worker.fcm_device_token != None,Worker.fcm_device_token.isnot(''))), True)],
        #         else_=False))
            # and_(Worker.fcm_device_token != None,Worker.fcm_device_token.isnot('')))

        workers = self._workers.get_all_with_join(
            Worker,
            WorkerCycle,
            Worker.id == WorkerCycle.worker_id,
            and_(Worker.fcm_device_token != None,Worker.fcm_device_token.isnot(''))
        )
        # ,
        #     or_(WorkerCycle is None, WorkerCycle.is_completed == True)

        return workers


    def is_pool_of_workers_available(self, model_name, model_version):
        """
        This function checks whether there are enough workers available in the database that fulfills server configuration min_workers
        parameter
        """
        kwargs = {"name": model_name}
        if model_version is not None:
            kwargs["version"] = model_version

        server_config, _ = process_manager.get_configs(**kwargs)

        min_workers = server_config.get("min_workers", 2)
        workers_in_db = len(self.get_all_workers())

        logging.info(f"workers in db {workers_in_db}")
        logging.info(f"required workers {min_workers}")
        #
        return (
            True
            if (
                workers_in_db >= min_workers
            )
            else False
        )

    def reset_worker_online_status(self):
        self._workers.modify(
            values={"is_online": False}
        )

    def get_registered_worker_push_ids(self, model_name, model_version, _cycle, check_online_status=False):
        are_workers_available = self.is_pool_of_workers_available(model_name, model_version)
        workers_for_cycle = []
        workers_device_tokens = []
        _fl_process = process_manager.first(name=model_name, version=model_version)

        if are_workers_available:
            # Find all those workers who are assigned to any cycle and that cycle has not yet completed
            all_workers_in_any_cycle = self.get_all_workers_in_any_cycle()
            for w in all_workers_in_any_cycle:
                if check_online_status == True:
                    _assigned = None
                    if w.WorkerCycle is None or (w.WorkerCycle is not None and w.WorkerCycle.is_completed == True):
                        # Check if already exists a relation between the worker and the cycle.
                        if (w.WorkerCycle is not None) and (_cycle is not None) and (_cycle.id == w.WorkerCycle.cycle_id):
                            _assigned = True
                        else:
                            _assigned = False


                    if (_assigned is not None and _assigned == True) or (check_online_status == True and w.Worker.is_online == False):
                        continue

                workers_for_cycle.append(w.Worker)
                workers_device_tokens.append(w.Worker.fcm_device_token)

            workers_device_tokens = list(dict.fromkeys(workers_device_tokens))

        return workers_device_tokens, workers_for_cycle

    def get_registered_worker_push_ids_controlled(self, model_name, model_version, _cycle, check_online_status=False):
        all_workers_in_any_cycle = self.get_all_workers()
        workers_for_cycle = []
        workers_device_tokens = []

        for w in all_workers_in_any_cycle:
            workers_for_cycle.append(w)
            workers_device_tokens.append(w.fcm_device_token)

        workers_device_tokens = list(dict.fromkeys(workers_device_tokens))
        unique_workers = []
        my_dict = {}
        for token in workers_device_tokens:
            my_dict[token] = False

        for token in workers_device_tokens:
            for worker in workers_for_cycle:
                if token == worker.fcm_device_token and my_dict[token] == False:
                    unique_workers.append(worker)
                    my_dict[token] = True

        return workers_device_tokens, unique_workers
