import datetime
from api.configDB import *

class Camera(Document):
    name = StringField(required=True, default='')
    url = StringField(required=True, default='')
    area_id = ObjectIdField(required=True)
    violation_type = ObjectIdField(required=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow())
    updated_at = DateTimeField(default=datetime.datetime.utcnow())