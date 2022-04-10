from api.models.image_model import Image
from api.models.camera_model import Camera
from api.models.violation_model import Violation
from api.models.violation_type_model import ViolationType
from api.models.object_information_model import ObjectInformation
from rest_framework_mongoengine import serializers as serializer_mongoengine

class ImageSerializer(serializer_mongoengine.DocumentSerializer):
    class Meta:
        model = Image
        fields = '__all__'

class CameraSerializer(serializer_mongoengine.DocumentSerializer):
    class Meta:
        model = Camera
        fields = '__all__'

class ViolationSerializer(serializer_mongoengine.DocumentSerializer):
    class Meta:
        model = Violation
        fields = '__all__'

class ViolationTypeSerializer(serializer_mongoengine.DocumentSerializer):
    class Meta:
        model = ViolationType
        fields = '__all__'

class ObjectClassSerializer(serializer_mongoengine.DocumentSerializer):
    class Meta:
        model = ObjectInformation
        fields = '__all__'