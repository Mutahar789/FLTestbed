package org.openmined.syft.demo.fcm


open class PushEvent(var eventType: PushEventType)

enum class PushEventType(val value: String)
{
    CYCLE_REQUEST("cycle_request"),
    CYCLE_COMPLETED("cycle_completed"),
    PROCESS_COMPLETED("process_completed"),
    TYPE_WORKER_STATUS ("worker_online_status")
}
