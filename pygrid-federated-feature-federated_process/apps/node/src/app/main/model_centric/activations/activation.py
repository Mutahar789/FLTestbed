from ... import BaseModel, db


class Activation(BaseModel):
    """Activation table.

    Columns:
        id (Integer, Primary Key): Cycle ID.
        worker_id
        avg_activations
        cycles_for_avg

    """

    __tablename__ = "model_centric_activations"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    avg_activations = db.Column(db.LargeBinary)
    worker_id = db.Column(db.String(255))
    cycles_for_avg = db.Column(db.Integer)
    is_slow = db.Column(db.Boolean(), default=False)

    def __str__(self):
        return f"< Activations id : {self.id}, worker_id: {self.worker_id}, cycles_for_avg: {self.cycles_for_avg}>"
