# PyGrid imports
# Generic imports
from .. import db

from sqlalchemy import func
from sqlalchemy import text

class Warehouse:
    def __init__(self, schema):
        self._schema = schema

    def register(self, **kwargs):
        """Register e  new object into the database.

        Args:
            parameters : List of object parameters.
        Returns:
            object: Database Object
        """
        _obj = self._schema(**kwargs)
        db.session.add(_obj)
        db.session.commit()

        return _obj

    def register_all(self, instances):
        """Register e  new objects into the database.

        Args:
            parameters : List of object instances to store in database.
        Returns:
            object: Database Object
        """
        db.session.add_all(instances)
        db.session.commit()

    def query(self, **kwargs):
        """ Query db objects filtering by parameters
            Args:
                parameters : List of parameters used to filter.
        """
        objects = self._schema.query.filter_by(**kwargs).all()
        return objects

    def count(self, **kwargs):
        """Query and return the count.

        Args:
            parameters: List of parameters used to filter.
        Return:
            count: Count of object instances.
        """
        query = db.session.query(func.count(self._schema.id)).filter_by(**kwargs)
        return int(query.scalar())

    def first(self, **kwargs):
        """Query and return the first occurrence.

        Args:
            parameters: List of parameters used to filter.
        Return:
            object: First object instance.
        """

        return self._schema.query.filter_by(**kwargs).first()

    def last(self, **kwargs):
        """Query and return the last occurrence.

        Args:
            parameters: List of parameters used to filter.
        Return:
            object: Last object instance.
        """
        return (
            self._schema.query.filter_by(**kwargs)
            .order_by(self._schema.id.desc())
            .first()
        )

    def contains(self, **kwargs):
        """Check if the object id already exists in the database.

        Args:
            id: Object ID.
        """
        return self.first(**kwargs) != None

    def delete(self, **kwargs):
        """Delete an object from the database.

        Args:
            parameters: Parameters used to filter the object.
        """
        object_to_delete = self.query(**kwargs)
        db.session.delete(object_to_delete)
        db.session.commit()

    def modify(self, query=None, values=None):
        """Modifies one or many records."""
        if query != None:
            self._schema.query.filter_by(**query).update(values)
        else:
            self._schema.query.update(values)
        db.session.commit()

    def update(self):
        db.session.commit()

    def get_all(self, arg=None):
        return self._schema.query.filter(arg).all()

    def get_all_with_join(self, table_left, table_right, join_condition, filter_condition=None):
        return db.session.query(table_left, table_right).outerjoin(table_right, join_condition).filter(filter_condition).all()

    def delete_all_where(self,  **kwargs):
        self._schema.query.filter_by(**kwargs).delete()
        db.session.commit()