from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('mlapp.urls')),  # Include app URLs
]

# Serve media files (uploads) even when DEBUG=False for this demo deployment
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
