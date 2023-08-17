# from django import forms

from django.forms import ModelForm
from .models import FileUpload

class FileUploadForm(ModelForm):
    class Meta:
        model = FileUpload
        fields = ['apt_name', 'docfile']

