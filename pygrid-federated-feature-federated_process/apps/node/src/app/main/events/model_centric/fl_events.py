# Standard python imports
import base64
import json
import traceback
import uuid
import logging
from binascii import unhexlify
import time

from ...core.codes import CYCLE, MODEL_CENTRIC_FL_EVENTS, MSG_FIELD, RESPONSE_MSG
from ...core.exceptions import (
    CycleNotFoundError,
    MaxCycleLimitExceededError,
    PyGridError,
)
from ...model_centric.auth.federated import verify_token
from ...model_centric.controller import processes
from ...model_centric.processes import process_manager
from ...model_centric.workers import worker_manager
from ...model_centric.fcm import fcm_push_manager
from ...model_centric.fcm.fcm_push_manager import PUSH_TYPES
from ...model_centric.tasks.cycle import complete_cycle, run_task_once

# Local imports
# Local imports
from ..socket_handler import SocketHandler

# Singleton socket handler
handler = SocketHandler()


def host_federated_training(message: dict, socket=None) -> dict:
    """This will allow for training cycles to begin on end-user devices.

    Args:
        message : Message body sent by some client.
        socket: Socket descriptor.
    Returns:
        response : String response to the client
    """
    data = message[MSG_FIELD.DATA]
    response = {}

    try:


        # Retrieve JSON values
        serialized_model = unhexlify(
            data.get(MSG_FIELD.MODEL, None).encode()
        )  # Only one

        # print("===================>>>> Model before serialization ", serialized_model)


        serialized_client_plans = {
            k: unhexlify(v.encode()) for k, v in data.get(CYCLE.PLANS, {}).items()
        }  # 1 or *

        logging.info("================================== after serialized client plans")


        serialized_client_protocols = {
            k: unhexlify(v.encode()) for k, v in data.get(CYCLE.PROTOCOLS, {}).items()
        }  # 0 or *
        serialized_avg_plan = unhexlify(
            data.get(CYCLE.AVG_PLAN, None).encode()
        )  # Only one
        client_config = data.get(CYCLE.CLIENT_CONFIG, None)  # Only one
        server_config = data.get(CYCLE.SERVER_CONFIG, None)  # Only one

        # Create a new FL Process
        processes.create_process(
            model=serialized_model,
            client_plans=serialized_client_plans,
            client_protocols=serialized_client_protocols,
            server_averaging_plan=serialized_avg_plan,
            client_config=client_config,
            server_config=server_config,
        )
        response[CYCLE.STATUS] = RESPONSE_MSG.SUCCESS
    except Exception as e:  # Retrieve exception messages such as missing JSON fields.
        response[RESPONSE_MSG.ERROR] = str(e) + traceback.format_exc()

    response = {
        MSG_FIELD.TYPE: MODEL_CENTRIC_FL_EVENTS.HOST_FL_TRAINING,
        MSG_FIELD.DATA: response,
    }

    return response


def assign_worker_id(message: dict, socket=None) -> dict:
    """New workers should receive a unique worker ID after authenticate on
    PyGrid platform.

    Args:
        message : Message body sended after token verification.
        socket: Socket descriptor.
    Returns:
        response : String response to the client
    """
    response = {}

    # Create a new worker instance and bind it with the socket connection.
    try:
        # Create new worker id
        worker_id = str(uuid.uuid4())

        # Create a link between worker id and socket descriptor
        handler.new_connection(worker_id, socket)

        # Create worker instance
        worker_manager.create(worker_id)

        requires_speed_test = True

        response[CYCLE.STATUS] = RESPONSE_MSG.SUCCESS
        response[MSG_FIELD.WORKER_ID] = worker_id

    except Exception as e:  # Retrieve exception messages such as missing JSON fields.
        response[CYCLE.STATUS] = RESPONSE_MSG.ERROR
        response[RESPONSE_MSG.ERROR] = str(e)

    return response


def requires_speed_test(model_name, model_version):

    kwargs = {"name": model_name}
    if model_version is not None:
        kwargs["version"] = model_version

    server_config, _ = process_manager.get_configs(**kwargs)

    #
    return (
        True
        if (
            server_config.get("minimum_upload_speed", None) is not None
            or server_config.get("minimum_download_speed", None) is not None
        )
        else False
    )


def authenticate(message: dict, socket=None) -> dict:
    """Check the submitted token and assign the worker a new id.

    Args:
        message : Message body sended by some client.
        socket: Socket descriptor.
    Returns:
        response : String response to the client
    """
    data = message.get("data")
    response = {}

    try:
        _auth_token = data.get("auth_token")
        model_name = data.get("model_name", None)
        model_version = data.get("model_version", None)

        verification_result = verify_token(_auth_token, model_name, model_version)

        if verification_result["status"] == RESPONSE_MSG.SUCCESS:
            response = assign_worker_id({"auth_token": _auth_token}, socket)
            # check if requires speed test
            response[MSG_FIELD.REQUIRES_SPEED_TEST] = requires_speed_test(
                model_name, model_version
            )


#            trigger_cycle_start_event(model_name, model_version)

            # run_task_once("trigger_cycle_event", trigger_cycle_start_event, model_name, model_version, PUSH_TYPES.TYPE_REQUEST_FOR_CYLE)



        else:
            response[RESPONSE_MSG.ERROR] = verification_result["error"]

    except Exception as e:
        response[RESPONSE_MSG.ERROR] = str(e) + "\n" + traceback.format_exc()

    response = {
        MSG_FIELD.TYPE: MODEL_CENTRIC_FL_EVENTS.AUTHENTICATE,
        MSG_FIELD.DATA: response,
    }

    return response




def cycle_request(message: dict, socket=None) -> dict:
    """This event is where the worker is attempting to join an active federated
    learning cycle.

    Args:
        message : Message body sent by some client.
        socket: Socket descriptor.
    Returns:
        response : String response to the client
    """
    data = message[MSG_FIELD.DATA]
    response = {}

    try:
        logging.info(data)
        # Retrieve JSON values
        worker_id = data.get(MSG_FIELD.WORKER_ID, None)
        name = data.get(MSG_FIELD.MODEL, None)
        version = data.get(CYCLE.VERSION, None)
        cycle_start_request_key = data.get(CYCLE.CYCLE_START_REQUEST_KEY, None)

        # Retrieve the worker
        worker = worker_manager.get(id=worker_id)

        # Request fields to worker's DB fields mapping
        fields_map = {
            CYCLE.PING: "ping",
            CYCLE.DOWNLOAD: "avg_download",
            CYCLE.UPLOAD: "avg_upload",
        }
        requires_speed_fields = requires_speed_test(name, version)

        # Check and save connection speed to DB
        for request_field, db_field in fields_map.items():
            if request_field in data:
                value = data.get(request_field)
                if not isinstance(value, (float, int)) or value < 0:
                    raise PyGridError(
                        f"'{request_field}' needs to be a positive number"
                    )
                setattr(worker, db_field, float(value))
            elif requires_speed_fields:
                # Require fields to present when FL model has speed req's
                raise PyGridError(f"'{request_field}' is required")

        worker_manager.update(worker)  # Update database worker attributes


        # The last time this worker was assigned for this model/version.
        last_participation = processes.last_cycle(worker_id, name, version)

        # Assign
        response = processes.assign(name, version, worker, last_participation, cycle_start_request_key)
    except CycleNotFoundError:
        # Nothing to do
        response[CYCLE.STATUS] = CYCLE.REJECTED
    except MaxCycleLimitExceededError as e:
        response[CYCLE.STATUS] = CYCLE.REJECTED
        response[MSG_FIELD.MODEL] = e.name
    except Exception as e:
        print("Exception: ", str(e))
        response[CYCLE.STATUS] = CYCLE.REJECTED
        response[RESPONSE_MSG.ERROR] = str(e) + traceback.format_exc()

    response = {
        MSG_FIELD.TYPE: MODEL_CENTRIC_FL_EVENTS.CYCLE_REQUEST,
        MSG_FIELD.DATA: response,
    }
    return response


def report(message: dict, socket=None) -> dict:
    """This method will allow a worker that has been accepted into a cycle and
    finished training a model on their device to upload the resulting model
    diff.

    Args:
        message : Message body sent by some client.
        socket: Socket descriptor.
    Returns:
        response : String response to the client
    """
    response = {}

    timeInSeconds = time.time()
    model_report_end_time = round(time.time() * 1000)
    data = message[MSG_FIELD.DATA]
    # logging.info(data)
    # exit()


    try:
        worker_id = data.get(MSG_FIELD.WORKER_ID, None)
        request_key = data.get(CYCLE.KEY, None)
        test_accuracy_list = data.get("test_accuracy_list", None)
        trained_epochs = data.get("trained_epochs", None)

        model_download_time = data.get("model_download_time", None)
        model_report_time = data.get("model_report_time", None)
        model_download_size = data.get("model_download_size", None)
        model_report_size = data.get("model_report_size", None)

        logging.info(f">>>>>>>>>>>>>>>>>>>>>model_report_time {model_report_time}, model_download_time {model_download_time}")



        trained_epochs = data.get("trained_epochs", None)

        num_samples = data.get("train_num_samples", None)
        model_size =  data.get(CYCLE.MODEL_SIZE, None)
        avg_epoch_time = data.get(CYCLE.AVG_EPOCH_TIME, None)
        total_train_time = data.get(CYCLE.TOTAL_TRAIN_TIME, None)
        cycle_id = data.get(CYCLE.CYCLE_ID, None)

        # It's simpler for client (and more efficient for bandwidth) to use base64
        # diff = unhexlify()
        diff = base64.b64decode(data.get(CYCLE.DIFF, None).encode())

        _worker = worker_manager.get(id=worker_id)

        kwargs = {"name": "mnist"}
        kwargs["version"] = "1.0"

        server_config, _ = process_manager.get_configs(**kwargs)
        optimizer = server_config.get("optimizer", "hasaas")

        if data.get(CYCLE.AVG_ACTIVATIONS, None) is not None:
            avgActivations =  base64.b64decode(data.get(CYCLE.AVG_ACTIVATIONS, None).encode())
            # fl_process_id: int, activations: bin, worker_id: str, cycles_for_avg: int
            processes.save_activations(1, avgActivations, _worker, 2)

        processes.save_metric(cycle_id, worker_id, test_accuracy_list, model_size, avg_epoch_time, total_train_time, int(_worker.is_slow), trained_epochs, model_download_time, model_report_time, model_download_size, model_report_size)

        # Submit model diff and run cycle and task async to avoid block report request
        # (for prod we probably should be replace this with Redis queue + separate worker)
        if optimizer != "fedavg":
            processes.submit_diff(worker_id, request_key, diff, num_samples, _worker)
        else:
            if not _worker.is_slow:
                processes.submit_diff(worker_id, request_key, diff, num_samples, _worker)

        # model_report_time = model_report_end_time - model_report_start_time
        response[CYCLE.STATUS] = RESPONSE_MSG.SUCCESS
    except Exception as e:  # Retrieve exception messages such as missing JSON fields.
        response[RESPONSE_MSG.ERROR] = str(e) + traceback.format_exc()

    response = {
        MSG_FIELD.TYPE: MODEL_CENTRIC_FL_EVENTS.REPORT,
        MSG_FIELD.DATA: response,
    }
    return response


def process_report_stats(message: dict, socket=None) -> dict:
    """This method will allow a worker that has been accepted into a cycle and
    finished training a model on their device to upload the resulting model
    diff.

    Args:
        message : Message body sent by some client.
        socket: Socket descriptor.
    Returns:
        response : String response to the client
    """
    data = message[MSG_FIELD.DATA]
    # logging.info(data)
    # exit()
    response = {}

    try:
        worker_id = data.get(MSG_FIELD.WORKER_ID, None)
        test_samples = data.get(CYCLE.TEST_SAMPLES, None)
        # test_accuracy = data.get(CYCLE.TEST_ACCURACY, None)
        # test_loss = data.get(CYCLE.TEST_LOSS, None)
        test_accuracy_list = data.get("test_accuracy_list", None)
        model_size =  data.get(CYCLE.MODEL_SIZE, None)
        avg_epoch_time = data.get(CYCLE.AVG_EPOCH_TIME, None)
        total_train_time = data.get(CYCLE.TOTAL_TRAIN_TIME, None)
        cycle_id = data.get(CYCLE.CYCLE_ID, None)
        logging.info(test_accuracy_list)

        _worker = worker_manager.get(id=worker_id)
        processes.save_metric(cycle_id, worker_id, test_accuracy_list, model_size, avg_epoch_time, total_train_time, int(_worker.is_slow), 10, "10", "10", "10", "10")

        run_task_once("average_diff", average_diff)

        response[CYCLE.STATUS] = RESPONSE_MSG.SUCCESS
    except Exception as e:  # Retrieve exception messages such as missing JSON fields.
        response[RESPONSE_MSG.ERROR] = str(e) + traceback.format_exc()
        print("Exception: ", str(e))

    # exit()
    response = {
        MSG_FIELD.TYPE: MODEL_CENTRIC_FL_EVENTS.REPORT_STATS,
        MSG_FIELD.DATA: response,
    }
    return response

def average_diff() :


   processes._average_plan_diffs()

def save_fcm_device_registration_token(message: dict, socket=None, cycle_manager=None) -> dict:
    data = message[MSG_FIELD.DATA]
    response = {}

    try:
        # Retrieve JSON values
        worker_id = data.get(MSG_FIELD.WORKER_ID, None)
        model_name = data.get(MSG_FIELD.MODEL, None)
        model_version = data.get(CYCLE.VERSION, None)

        kwargs = {"name": model_name}
        if model_version is not None:
            kwargs["version"] = model_version

        server_config, _ = process_manager.get_configs(**kwargs)
        worker_participation_mode = server_config.get("worker_participation_mode", 1)

        _fl_process = process_manager.first(name=model_name, version=model_version)

        # Retrieve the last cycle used by this fl process/ version
        _cycle = cycle_manager.last(_fl_process.id, None)

        worker = worker_manager.get(id=worker_id)
        worker.fcm_device_token = data.get("fcm_device_token", None)
        worker.ram = data.get("ram_size", None)
        worker.cpu_cores = data.get("cpu_cores", None)

        if worker.ram < 3:
            worker.is_slow = True
        else:
            worker.is_slow = False

        # worker.is_slow = True
        worker.is_online = True
        worker_manager.update(worker)  # Update database worker attributes

        logging.info(f"_cycle.cycle_start_key {_cycle.cycle_start_key}")

        # if (not _cycle.cycle_start_key) or worker_participation_mode == 1:
        fcm_push_manager.trigger_event(model_name, model_version, PUSH_TYPES.TYPE_REQUEST_FOR_CYLE, _cycle, cycle_manager, _fl_process)

        # Assign
        response[CYCLE.STATUS] = "token has been saved"
    except Exception as e:
        print("Exception: ", str(e))
        response[CYCLE.STATUS] = CYCLE.REJECTED
        response[RESPONSE_MSG.ERROR] = str(e) + traceback.format_exc()

    response = {
        MSG_FIELD.TYPE: "model-centric/save-fcm-token",
        MSG_FIELD.DATA: response,
    }
    return response

def update_worker_online_status(message: dict, socket=None) -> dict:
    logging.info(f"================== update_worker_online_status {message}" )
    data = message[MSG_FIELD.DATA]
    response = {}

    try:
        worker_id = data.get(MSG_FIELD.WORKER_ID, None)
        fcm_push_token = data.get(MSG_FIELD.FCM_PUSH_TOKEN, None)
        worker = worker_manager.get(id=worker_id)
        worker.is_online = True
        worker.fcm_device_token = fcm_push_token
        worker_manager.update(worker)  # Update database worker attributes


        # Assign
        response[CYCLE.STATUS] = "status updated with success."
    except Exception as e:
        print("Exception: ", str(e))
        response[CYCLE.STATUS] = "Error"
        response[RESPONSE_MSG.ERROR] = str(e) + traceback.format_exc()

    response = {
        MSG_FIELD.TYPE: "model-centric/update-worker-status",
        MSG_FIELD.DATA: response,
    }
    return response


def upload_file(message: dict, socket=None):
    return message
