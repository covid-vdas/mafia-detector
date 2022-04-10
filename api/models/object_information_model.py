import datetime
from api.configDB import *

class ObjectInformation(Document):
    cardinality = IntField(required=True, default=0)
    name = StringField(required=True, default='')
    created_at = DateTimeField(default=datetime.datetime.utcnow())
    updated_at = DateTimeField(default=datetime.datetime.utcnow())