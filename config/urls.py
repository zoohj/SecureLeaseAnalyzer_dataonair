"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import include, path
from django.conf.urls.static import static
from django.conf import settings
from app.views import file_upload, success_upload, test # Import the view function
from app import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('fileupload/', file_upload, name="file_upload"), # 등본파일 업로드
    path('success_upload/', success_upload, name='success_upload'), # 결과
    path('test/', test, name='test'), # 결과


] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
