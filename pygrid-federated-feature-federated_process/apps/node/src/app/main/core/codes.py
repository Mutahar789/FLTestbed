class MSG_FIELD:
    REQUEST_ID = "request_id"
    TYPE = "type"
    DATA = "data"
    WORKER_ID = "worker_id"
    MODEL = "model"
    MODEL_ID = "model_id"
    ALIVE = "alive"
    ALLOW_DOWNLOAD = "allow_download"
    ALLOW_REMOTE_INFERENCE = "allow_remote_inference"
    MPC = "mpc"
    PROPERTIES = "model_properties"
    SIZE = "model_size"
    SYFT_VERSION = "syft_version"
    REQUIRES_SPEED_TEST = "requires_speed_test"
    USERNAME_FIELD = "username"
    PASSWORD_FIELD = "password"
    FCM_PUSH_TOKEN = "fcm_push_token"
    STATS = "stats"
    NUM_SAMPLES = "num_samples"
    ACCURACY = "accuracy"

class CONTROL_EVENTS(object):
    SOCKET_PING = "socket-ping"


class WEBRTC_EVENTS(object):
    PEER_LEFT = "webrtc: peer-left"
    INTERNAL_MSG = "webrtc: internal-message"
    JOIN_ROOM = "webrtc: join-room"


class MODEL_CENTRIC_FL_EVENTS(object):
    HOST_FL_TRAINING = "model-centric/host-training"
    REPORT = "model-centric/report"
    REPORT_STATS = "model-centric/report_stats"
    AUTHENTICATE = "model-centric/authenticate"
    CYCLE_REQUEST = "model-centric/cycle-request"
    SAVE_FCM_TOKEN = "model-centric/save-fcm-token"
    SEND_PUSH_EVENT = "model-centric/send_push_event"
    UPDATE_WORKER_ONLINE_STATUS = "model-centric/update-worker-status"
    UPLOAD_STATS = "model-centric/upload_stats"

class USER_EVENTS(object):
    GET_ALL_USERS = "list-users"
    GET_SPECIFIC_USER = "list-user"
    SEARCH_USERS = "search-users"
    PUT_EMAIL = "put-email"
    PUT_PASSWORD = "put-password"
    PUT_ROLE = "put-role"
    PUT_GROUPS = "put-groups"
    DELETE_USER = "delete-user"
    SIGNUP_USER = "signup-user"
    LOGIN_USER = "login-user"


class ROLE_EVENTS(object):
    CREATE_ROLE = "create-role"
    GET_ROLE = "get-role"
    GET_ALL_ROLES = "get-all-roles"
    PUT_ROLE = "put-role"
    DELETE_ROLE = "delete-role"


class GROUP_EVENTS(object):
    CREATE_GROUP = "create-group"
    GET_GROUP = "get-group"
    GET_ALL_GROUPS = "get-all-groups"
    PUT_GROUP = "put-group"
    DELETE_GROUP = "delete-group"


class CYCLE(object):
    STATUS = "status"
    KEY = "request_key"
    PING = "ping"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    VERSION = "version"
    PLANS = "plans"
    PROTOCOLS = "protocols"
    CLIENT_CONFIG = "client_config"
    SERVER_CONFIG = "server_config"
    TIMEOUT = "timeout"
    DIFF = "diff"
    AVG_ACTIVATIONS = "avgActivations"
    AVG_PLAN = "averaging_plan"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CYCLE_ID = "cycle_id"
    CYCLE_START_REQUEST_KEY = "cycle_start_request_key"
    NUM_SAMPLES = "num_samples"
    TEST_SAMPLES = "test_samples"
    TEST_ACCURACY = "test_accuracy"
    TEST_ACCURACY = "test_accuracy_list"
    TEST_LOSS = "test_loss"
    MODEL_SIZE = "model_size"
    AVG_EPOCH_TIME = "avg_epoch_train_time"
    TOTAL_TRAIN_TIME = "total_train_time"


class RESPONSE_MSG(object):
    ERROR = "error"
    SUCCESS = "success"
