from django.urls import path

from api.views import DetectorView
from api.views.stream_views import StreamView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('detector/', DetectorView.as_view()),
    path('stream/', StreamView.as_view()),
]

if settings.DEBUG:
  urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)