# External imports
import syft as sy

# Local imports
from ... import BaseModel, db


class ModelMask(BaseModel):
    __tablename__ = "model_centric_model_mask"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    mask = db.Column(db.String)
    model_id = db.Column(
        db.Integer, db.ForeignKey("model_centric_model.id"), unique=True
    )

    def __str__(self):
        return f"<ModelMask  id: {self.id}, mask: {self.mask}>"