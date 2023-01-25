# Metrics module imports
import logging

# PyGrid modules
from ...core.warehouse import Warehouse
from ...core.codes import CYCLE, MSG_FIELD, RESPONSE_MSG
from .metrics import Metrics
import os
from ..cycles.cycle import  Cycle
from ..models import model_manager
from ..cycles import cycle_manager
from ..processes import process_manager
class MetricsManager:
    def __init__(self):
        self._metrics = Warehouse(Metrics)
        self._cycles = Warehouse(Cycle)

    def save_metric(self, cycle_id, worker_id, test_accuracy_list, model_size, avg_epoch_time, total_train_time, is_slow, trained_epochs, model_download_time, model_report_time, model_download_size, model_report_size):
        # _metric = self._metrics.register(
        #     cycle_id=cycle_id, worker_id = worker_id, accuracy = test_accuracy, loss = test_loss, num_samples = test_samples, model_size = model_size, avg_epoch_time = avg_epoch_time, total_train_time = total_train_time,
        #     is_slow=is_slow
        # )

        metrics_list = []
        counter = 1
        for accuracy in test_accuracy_list:
            test_accuracy = accuracy.get("test_accuracy")
            num_samples = accuracy.get("num_samples")
            metric = Metrics(cycle_id=cycle_id,
                             worker_id = f"{worker_id}_{counter}",
                             accuracy = test_accuracy,
                             num_samples = num_samples,
                             model_size = model_size,
                             avg_epoch_time = avg_epoch_time,
                             total_train_time = total_train_time,
                             is_slow=is_slow,
                             trained_epochs=trained_epochs,
                             model_download_time=float(model_download_time),
                             model_download_size=int(model_download_size))
            counter += 1

            metrics_list.append(metric)

        self._metrics.register_all(metrics_list)

        if model_report_time != None and len(model_report_time) > 0 and int(cycle_id) > 1:
            # splitArr = model_report_time.split(":")
            # report_cycle_id = int(splitArr[0])
            # report_time = float(splitArr[1])


            _metric = self._metrics.first(cycle_id=int(cycle_id) - 1, worker_id=f"{worker_id}_1")
            if _metric != None:
                _metric.model_report_time = float(model_report_time)
                _metric.model_report_size = int(model_report_size)
                self._metrics.update()



    def store(self, stats_list):
        """Create a new federated learning cycle.

        Args:
            fl_process_id: FL Process's ID.
            version: Version (?)
            cycle_time: Remaining time to finish this cycle.
        Returns:
            fd_cycle: Cycle Instance.
        """

        instances = []

        metrics_list = []

        for message in stats_list:
            cycle_id = message.get(CYCLE.CYCLE_ID)
            worker_id = message.get(MSG_FIELD.WORKER_ID)
            accuracy = message.get(MSG_FIELD.ACCURACY)
            num_samples = message.get(MSG_FIELD.NUM_SAMPLES)

            metric = Metrics(cycle_id=cycle_id,worker_id=worker_id,accuracy=accuracy,num_samples=num_samples)

            metrics_list.append(metric)

        self._metrics.register_all(metrics_list)

    # def get_all_stats(self):
    #     metrics =  self._metrics.query()
    #
    #     response = []
    #     for metric in metrics:
    #         response.append(metric.as_dict())
    #
    #     self.create_csv(metrics)
    #
    #     return response

    def query(self, **kwargs):
        return self._metrics.query(**kwargs)
    def get_all_stats(self):

        # import numpy as np
        #
        # all_checkpoints = model_manager.get_all()
        #
        # print("all_checkpoints: ", len(all_checkpoints))
        # for checkpoint in all_checkpoints:
        #     model_params = model_manager.unserialize_model_params(checkpoint.value)
        #
        #     list_model_params = []
        #     for param in model_params:
        #         list_model_params.append(param.data.numpy())
        #
        #     np_weights = np.array(list_model_params)
        #     np.save(f'weights_for_round_{checkpoint.id}', np_weights)
        #
        # return
        # _last_checkpoint = model_manager.load(model_id=1).value
        # model_params = model_manager.unserialize_model_params(_last_checkpoint)
        #
        # list_model_params = []
        # for param in model_params:
        #     list_model_params.append(param.data.numpy())
        #
        # np_weights = np.array(list_model_params)
        # np.save(f'weights_for_round_{110}', np_weights)

        # model_name = "mnist"
        # model_version = "1.0"
        # _fl_process = process_manager.first(name=model_name, version=model_version)
        # # Retrieve the last cycle used by this fl process/ version
        # _cycle = cycle_manager.last(_fl_process.id, None)
        #
        # kwargs = {"name": model_name}
        # if model_version is not None:
        #     kwargs["version"] = model_version
        #
        # server_config, _ = process_manager.get_configs(**kwargs)
        # cycle_manager._average_plan_diffs(server_config, _cycle)
        # exit()

        metrics = self._metrics.query()

        # result = cycle_manager.get_cycle_details()
        result = self._cycles.get_all_with_join(
            Metrics,
            Cycle,
            Metrics.cycle_id == Cycle.id
        )

        logging.info(f'============== result {result}')

        cycles = self._cycles.query(
        )

        cycle_times = {}
        count = 0
        for cycle in cycles:
            if cycle.completed_at is None:
                continue
            cycle_times[cycle.id] = (cycle.completed_at - cycle.start).seconds

        # print(cycle_times)

        # logging.info(f'============== cycles {cycles}')

        response = []
        for index, metric in enumerate(metrics):
            if metric.cycle_id not in cycle_times:
                continue
            metric.round_completion_time = cycle_times[metric.cycle_id]
            metrics_dict = metric.as_dict()
            metrics[index].round_completion_time = cycle_times[metric.cycle_id]
            metrics_dict['round_completion_time'] = cycle_times[metric.cycle_id]

            logging.info(f'============== metrics_dict {metrics_dict}')
            response.append(metrics_dict)

        self.create_csv(metrics)

        return response

    def create_csv(self, data):
        """
            Create CSV for metrics
        """

        logging.info(f"=============================== create_csv")
        file_basename = 'metrics_stat_testbed.csv'
        server_path = '../../examples/model-centric/data'
        w_file = open(os.path.join(server_path, file_basename), 'w')
        w_file.write('client_id,round_number,hierarchy,num_samples,set,accuracy,loss,model_size,avg_epoch_time,total_train_time,is_slow,round_completion_time,trained_epochs,model_download_time,model_report_time,model_download_size,model_report_size\n')

        for r in data:
            record = str(r)
            logging.info(f"================== row_as_string {type(r)}")
            w_file.write(record + '\n')

        w_file.close()

