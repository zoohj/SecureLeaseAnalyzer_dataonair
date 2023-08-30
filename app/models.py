from datetime import timezone
from django.db import models
import os

def upload_to(instance, filename):
    now = timezone.now()
    filename_base, filename_ext = os.path.splitext(filename)
    return f'uploads/{instance.apt_name}/{now.strftime("%Y%m%d%H%M%S")}{filename_ext}'

class FileUpload(models.Model):
    apt_name = models.CharField(max_length=20)
    docfile = models.FileField(upload_to='uploads/')  # media/uploads/ 폴더에 저장
