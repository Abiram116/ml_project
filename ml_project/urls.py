from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('mlapp.urls')),  # Include app URLs
]

# Add media URL configuration
if settings.DEBUG:  # Serve media files during development only
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
