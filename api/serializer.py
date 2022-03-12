from api.models.image_model import Image
from rest_framework_mongoengine import serializers as serializer_mongoengine

class ImageSerializer(serializer_mongoengine.DocumentSerializer):
    class Meta:
        model = Image
        fields = '__all__'