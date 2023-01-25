# Cycle module imports

from .activation import Activation
from ...core.warehouse import Warehouse
from ..workers.worker import Worker
from sqlalchemy import and_
class ActivationManager:
    def __init__(self):
        self._activations = Warehouse(Activation)

    def save_activations(self, fl_process_id: int, activations: bin, worker_id: str, cycles_for_avg: int, worker_status: bool):
        """Create a new federated learning cycle.

        Args:
            fl_process_id: FL Process's ID.
            version: Version (?)
            cycle_time: Remaining time to finish this cycle.
        Returns:
            fd_cycle: Cycle Instance.
        """
        _worker_cycle = self._activations.register(
            worker_id=worker_id, avg_activations=activations, cycles_for_avg=cycles_for_avg, is_slow=worker_status
        )

    def get_activations(self):
        return self._activations.get_all_with_join(
            Activation,
            Worker,
            Activation.worker_id == Worker.id,
            and_(Worker.fcm_device_token != None, Worker.fcm_device_token.isnot(''))
        )

