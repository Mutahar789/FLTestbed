from ... import BaseModel, db


class Metrics(BaseModel):
    """Metrics table.

    Columns:
        id (Integer, Primary Key): Auto increment
        cycle_id Cycle ID.
        worker_id String
        accuracy Float
        num_samples Int
    """

    __tablename__ = "model_centric_metrics"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cycle_id = db.Column(db.Integer)
    worker_id = db.Column(db.String(255))
    accuracy = db.Column(db.Float)
    loss = db.Column(db.Float)
    num_samples = db.Column(db.Integer)
    model_size = db.Column(db.Float)
    avg_epoch_time = db.Column(db.Float)
    total_train_time = db.Column(db.Float)
    is_slow = db.Column(db.Integer)
    round_completion_time = db.Column(db.Float)
    trained_epochs = db.Column(db.Integer)
    model_download_time = db.Column(db.Float)
    model_report_time  = db.Column(db.Float)
    model_download_size = db.Column(db.Integer)
    model_report_size = db.Column(db.Integer)

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    def __str__(self):
        # return f"< Metrics id : {self.id}, cycle_id: {self.cycle_id}, worker_id: {self.worker_id}, accuracy: {self.accuracy}, num_samples: {self.num_samples}"
        return f"{self.worker_id}, {self.cycle_id}, ,{self.num_samples}, test, {self.accuracy},{self.loss}, {self.model_size}, {self.avg_epoch_time}, {self.total_train_time}, {self.is_slow}, {self.round_completion_time}, {self.trained_epochs}, {self.model_download_time}, {self.model_report_time}, {self.model_download_size}, {self.model_report_size}"
