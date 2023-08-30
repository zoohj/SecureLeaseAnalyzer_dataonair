import os
from django.conf import settings
from django.shortcuts import render, redirect

from app.models import FileUpload
from .forms import FileUploadForm
from .clustering import analysis

import csv


def file_upload(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            apt_name = form.cleaned_data['apt_name']
            docfile = form.cleaned_data['docfile']
            form.save()  # 아파트명 및 등본 저장
            print(apt_name) # check
            print('파일 업로드 하는 곳')

            docfile_obj = request.FILES['docfile']

            percent= analysis(apt_name, docfile_obj)

            # make_data(request, apt_name, docfile)
            print( percent)
            
            return render(request, 'success_upload.html', {'percent': percent})
            # return redirect('success_upload') # Redirect to a success page
    else:   
        form = FileUploadForm()
    return render(request, 'home.html', {'form': form})


def success_upload(request):
    return render(request, 'success_upload.html')


def test(request):
    # percent = 2.25
    # percent_test = "{:.2%}".format(percent / 100)
    # print(percent_test)
    return render(request, 'success_upload.html')
    # return render(request, 'result_test.html', {'percent': percent_test})

