from ... import BaseModel, db
from sqlalchemy.orm import backref


class Worker(BaseModel):
    """Web / Mobile worker table.

    Columns:
        id (String, Primary Key): Worker's ID.
        ping (Float): Ping rate.
        avg_download (Float): Download rate.
        avg_upload (Float): Upload rate.
        worker_cycles (WorkerCycle): Relationship between workers and cycles (One to many).
    """

    __tablename__ = "model_centric_worker"

    id = db.Column(db.String(255), primary_key=True)
    ping = db.Column(db.Float)
    avg_download = db.Column(db.Float)
    avg_upload = db.Column(db.Float)
    fcm_device_token = db.Column(db.String)
    is_online = db.Column(db.Boolean(), default=False)
    is_slow = db.Column(db.Boolean(), default=False)
    worker_cycle = db.relationship("WorkerCycle", backref="worker")
    ram = db.Column(db.Float)
    cpu_cores = db.Column(db.Integer)

    def __str__(self):
        return f"<Worker id: {self.id}, is_slow : {self.is_slow}, is_online: {self.is_online}>"
