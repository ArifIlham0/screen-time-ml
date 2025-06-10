from django.contrib import admin
from django.urls import path, include
from screen_time_ml.views.intensity_classification_view import intensity_classification

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('classify-intensity', include('core.urls')),
    path('classify-intensity', intensity_classification),
]
