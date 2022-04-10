import datetime
from api.configDB import *

class Violation(Document):
    type_id = ObjectIdField(required=True)
    camera_id = StringField()
    class_id = ObjectIdField(required=True)
    image_id = ObjectIdField(required=True)
    distance = StringField()
    created_at = DateTimeField(default=datetime.datetime.utcnow())
    updated_at = DateTimeField(default=datetime.datetime.utcnow())
