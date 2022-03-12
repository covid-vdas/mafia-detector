import datetime
from api.configDB import *

class Image(Document):
    name = StringField(required=True)
    url = StringField(required=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow())
    updated_at = DateTimeField(default=datetime.datetime.utcnow())

