import sqlite3
from pyfcm import FCMNotification

from ..workers import worker_manager
from ..tasks.fcm import run_task_once
from ...core.codes import CYCLE, MODEL_CENTRIC_FL_EVENTS, MSG_FIELD, RESPONSE_MSG
from ...core.exceptions import CycleNotFoundError
from ..processes import process_manager
import random
import abc
import logging
import time
import uuid
import hashlib
import math
# from apscheduler.schedulers.background import BackgroundScheduler, BlockingScheduler

class FcmManager(metaclass=abc.ABCMeta):
    def __init__(self):
        self.push_service = FCMNotification(api_key="AAAAsdENP3E:APA91bHNWoH_kx72iYfUUIIsDgaUHV_QDqEbWTW3WuXVVH0q-E7VVmoVC-HFnQuMzcSfm6XJxy0X188ofrKpxha0Ky-jsr6CmdyNlzGDpLU5I5FlctGMJ6bBZgm51ydCzoyO0FScDQjB")

    def send_push(self, registration_ids, push_data):

        response_body = {}

        #logging.info(f"==============>>>>>>>> send_push {push_data}")


        result = self.push_service.notify_multiple_devices(registration_ids=registration_ids, 
            message_title="This is message title", 
            message_body="this is message_body",
            data_message=push_data)

        # def got_result(result):
        #     print(result)

        # df.addBoth(got_result)
        # reactor.run()

        print(result)
        
        response_body["status"] = "success"

        return response_body

    def send_push_event(self, device_tokens, push_type, push_data) -> dict:
        """Send push notification.

        Args:
            message : Message body sended by some client.
            socket: Socket descriptor.
        Returns:
            response : String response to the client
        """
        logging.info(f"Sending push notification for {push_type}")
        
        response = {}

        # push_data = {
        #     MSG_FIELD.TYPE: push_type,
        #     MSG_FIELD.DATA: data
        # }
        

        self.send_push(device_tokens, push_data)
        
        response = {
            MSG_FIELD.TYPE: MODEL_CENTRIC_FL_EVENTS.SEND_PUSH_EVENT,
            MSG_FIELD.DATA: {
                    CYCLE.STATUS: "SUCCESS"
                }
        }

        return  response

   
    def trigger_event(self, model_name, model_version, event_type, _cycle, cycle_manager, _fl_process):
        """
        Here whenever user connects with server, check with the database
        that are the minimum required workers available for training. If Yes then do the following:
        1. If synchronous and cycle is already started then do nothing as on each cycle completion we will send event to all connected
        workers to request for a cycle
        2. If asynchronous then send event to all those workers that are not part of current cycle to start request for a cycle.
        """

        # if _cycle is not None and _cycle.cycle_start_key is not None and len(_cycle.cycle_start_key) > 0:
        #     logging.info(f"============================>>>>>>>>>>>>>>>>>>>>>>> ALREADY RUNNING: {event_type} cycle id {_cycle.id} ")
        #     return

        # workers_device_tokens, _ = worker_manager.get_registered_worker_push_ids(model_name, model_version, _cycle)

        # push_data = {
        #     MSG_FIELD.TYPE: PUSH_TYPES.TYPE_WORKER_STATUS,
        #     MSG_FIELD.DATA: {CYCLE.STATUS: "SUCCESS"}
        # }
        #
        # self.send_push_event(workers_device_tokens, PUSH_TYPES.TYPE_WORKER_STATUS, push_data)

        # worker_manager.reset_worker_online_status()

        # from ...routes.model_centric.routes import scheduler
        # app = scheduler.app
        # with app.app_context():
        # kwargs = {'trigger': 'interval'}
        # kwargs["seconds"] = 2
        # kwargs["replace_existing"] = True
        # kwargs["args"] = [model_name, model_version, event_type, _cycle, cycle_manager, _fl_process]
        # scheduler.add_job('send_push_for_online_workers', send_push_for_online_workers, **kwargs)



        # with app.app_context():
        #     try:
        #         self.sched.remove_job(job_id="test_send_push_for_online_workers")
        #     except Exception as ex:
        #         print(ex)
        #
        #
        #     self.sched.add_job(func=lambda : self.send_push_for_online_workers(self.sched, app, model_name, model_version, event_type, _cycle, cycle_manager, _fl_process),
        #                         trigger='interval', seconds=2,
        #                         id="test_send_push_for_online_workers",
        #                         max_instances=10,
        #                         replace_existing=False)

        app = None
        self.send_push_for_online_workers(app, model_name, model_version, event_type, _cycle, cycle_manager,
                                          _fl_process)

        # run_task_once("send_push_for_online_workers", send_push_for_online_workers, self, model_name, model_version, event_type, _cycle, cycle_manager, _fl_process)


        return True

    def send_push_for_online_workers(self, app, model_name, model_version, event_type, cycle, cycle_manager, fl_process):
        """
            Wait for the worker acknowledgements, if enough workers are available then send them message for starting the next cycle.
        """
        logging.info(f"======================== send_push_for_online_workers")

        # with app.app_context():
        try:
            _fl_process = process_manager.first(name=model_name, version=model_version)
            _cycle = cycle_manager.last(_fl_process.id, None)

            # if _cycle.cycle_start_key is not None and len(_cycle.cycle_start_key) > 0:
            #     logging.info("============================>>>>>>>>>>>>>>>>>>>>>>> ALREADY RUNNING")
            #     return

            workers_device_tokens, workers = worker_manager.get_registered_worker_push_ids_controlled(model_name, model_version, _cycle, False)

            kwargs = {"name": model_name}
            if model_version is not None:
                kwargs["version"] = model_version

            server_config, _ = process_manager.get_configs(**kwargs)
            min_workers = server_config.get("min_workers", 2)
            droprate = server_config.get("drop_rate", 0)

            logging.info(f"================== send_push_for_online_workers {workers_device_tokens}")
            if len(workers_device_tokens) < min_workers:
                return

            # workers_device_tokens, workers = worker_manager.get_registered_worker_push_ids(model_name, model_version, _cycle, False)
            # logging.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> send_push_for_online_workers: required workers achieved")

            high_end_devices = [w for w in workers if w.is_slow == False]
            low_end_devices = [w for w in workers if w.is_slow == True]

            goal_count = server_config.get("goal_count", 2)

            slow_clients_count = math.ceil(goal_count * droprate)
            fast_clients_count = goal_count - slow_clients_count

            logging.info(
                f"================================= goal_count: {goal_count}, total_high {len(high_end_devices)}, total_low {len(low_end_devices)}")

            if fast_clients_count > len(high_end_devices):
                fast_clients_count = len(high_end_devices)

            if slow_clients_count > len(low_end_devices):
                slow_clients_count = len(low_end_devices)

            # user_high = random.choices(high_end_devices, k=fast_clients_count)#high_end_devices[0:fast_clients]
            # user_low = random.choices(low_end_devices, k=slow_clients_count)

            user_high = high_end_devices[0:fast_clients_count]  # random.choices(high_end_devices, k=fast_clients_count)#high_end_devices[0:fast_clients]
            user_low = low_end_devices[0:slow_clients_count]  # random.choices(low_end_devices,

            logging.info(
                f"================================= slow_clients {slow_clients_count}, fast_clients {fast_clients_count}")
            logging.info(
                f"================================= user_low {user_low} user_high {user_high}")

            random_selected_user_token = []

            for user in user_high:
                random_selected_user_token.append(user.fcm_device_token)

            for user in user_low:
                random_selected_user_token.append(user.fcm_device_token)

            data = {}
            data[CYCLE.STATUS]: "SUCCESS"
            cycle_start_request_key = self._generate_hash_key(uuid.uuid4().hex)
            data[CYCLE.CYCLE_START_REQUEST_KEY] = cycle_start_request_key

            push_data = {
                MSG_FIELD.TYPE: event_type,
                MSG_FIELD.DATA: {
                    CYCLE.STATUS: "SUCCESS",
                    CYCLE.CYCLE_START_REQUEST_KEY: cycle_start_request_key
                }
            }

            try:
                _current_cycle = cycle_manager.last(_fl_process.id, None)
                _current_cycle.cycle_start_key = cycle_start_request_key
                cycle_manager.update()

                # goal_count = server_config.get("goal_count", 2)
                # if min_workers == goal_count:
                #     random_selected_user_token = workers_device_tokens
                # else:
                #     random_selected_user_token = random.choices(workers_device_tokens, k=goal_count)

                self.send_push_event(random_selected_user_token, event_type, push_data)
            except CycleNotFoundError:
                logging.info("Cycle is not found")
                push_data = {
                    MSG_FIELD.TYPE: PUSH_TYPES.TYPE_FL_COMPLETED,
                    MSG_FIELD.DATA: {
                        CYCLE.STATUS: "SUCCESS"
                    }
                }
                self.send_push_event(workers_device_tokens, PUSH_TYPES.TYPE_FL_COMPLETED, push_data)
        except CycleNotFoundError:
            workers_device_tokens, _ = worker_manager.get_registered_worker_push_ids(model_name, model_version, None, True)
            logging.info("Cycle is not found")
            push_data = {
                MSG_FIELD.TYPE: PUSH_TYPES.TYPE_FL_COMPLETED,
                MSG_FIELD.DATA: {
                    CYCLE.STATUS: "SUCCESS"
                }
            }
            self.send_push_event(workers_device_tokens, PUSH_TYPES.TYPE_FL_COMPLETED, push_data)

    def _generate_hash_key(self, primary_key: str) -> str:
        """Generate SHA256 Hash to give access to the cycle.

        Args:
            primary_key : Used to generate hash code.
        Returns:
            hash_code : Hash in string format.
        """
        return hashlib.sha256(primary_key.encode()).hexdigest()
class PUSH_TYPES:
    TYPE_REQUEST_FOR_CYLE = "request_for_cycle"
    TYPE_CYCLE_COMPLETED = "cycle_completed"
    TYPE_INTERRUPT_CYCLE_COMPLETED = "interrupt_cycle_completed"
    TYPE_CYCLE_RESET = "cycle_completed"
    TYPE_FL_COMPLETED = "fl_completed"
    TYPE_WORKER_STATUS = "worker_online_status"

