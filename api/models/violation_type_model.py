import datetime
from api.configDB import *


class ViolationType(Document):
    name = StringField(required=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow())
    updated_at = DateTimeField(default=datetime.datetime.utcnow())
