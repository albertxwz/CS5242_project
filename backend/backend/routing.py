from django.urls import path
from backend.consumer import ImageConsumer

websocket_urlpatterns = [
    path('ws/get_image/', ImageConsumer.as_asgi()),
]
