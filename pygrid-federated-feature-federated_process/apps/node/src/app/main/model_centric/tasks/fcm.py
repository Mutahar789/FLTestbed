# Standard python imports
import logging
import traceback

# Local imports
from ... import executor


def run_task_once(name, func, *args):
    future = executor.futures._futures.get(name)
    logging.info("future: %s" % str(future))
    logging.info("futures count: %d" % len(executor.futures._futures))
    # prevent running multiple threads
    if future is None or future.done() is True:
        executor.futures.pop(name)
        try:
            executor.submit_stored(name, func, *args)
        except Exception as e:
            logging.error(
                "Failed to start new thread: %s %s" % (str(e), traceback.format_exc())
            )
    else:
        logging.warning(
            "Skipping %s execution because previous one is not finished" % name
        )


def trigger_cycle_event(model_name, model_version):
    logging.info("running trigger_cycle_event")
    try:
        # trigger_cycle_start_event(model_name, model_version)
        return True
    except Exception as e:
        logging.error(
            "Error in complete_cycle task: %s %s" % (str(e), traceback.format_exc())
        )
        return e

# def send_push_for_online_workers(fcm_manager, model_name, model_version, event_type, _cycle, cycle_manager, _fl_process):
#     logging.info("running send_push_for_online_workers")
#     try:
#         fcm_manager.send_push_for_online_workers(model_name, model_version, event_type, _cycle, cycle_manager, _fl_process)
#         return True
#     except Exception as e:
#         logging.error(
#             "Error in send_push_for_online_workers task: %s %s" % (str(e), traceback.format_exc())
#         )
#         return e
