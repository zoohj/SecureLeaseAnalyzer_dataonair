# from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect

# from app.forms import DocumentForm

from .forms import FileUploadForm
# from .models import FileUpload


# Create your views here.

#views
from django.shortcuts import render, redirect
from .forms import FileUploadForm

def file_upload(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            apt_name = form.cleaned_data['apt_name']
            form.save()  # Save the form data to the model
            return redirect('result')  # Redirect to a success page
    else:
        form = FileUploadForm()
    return render(request, 'home.html', {'form': form})

def result(request):
    return render(request, 'result.html')

def result(request):
    return render(request, 'home.html')
from django.shortcuts import render

def result(request):
    # 등기부등본 분석 및 결과 데이터 생성 로직
    # 예시 데이터, 실제 데이터에 맞게 수정 필요
    visualization_image_path = 'images/visualization.png'
    clustering_result = '클러스터링 결과 및 해석'
    probability = '확률 정보'

    return render(request, 'result.html', {
        'visualization_image_path': visualization_image_path,
        'clustering_result': clustering_result,
        'probability': probability
    })





# def file_upload(request):
#     if request.method == 'POST':
#         apt_name = request.POST['apt_name']
#         docfile= request.POST['docfile']
        
#         fileupload = FileUpload(
#             apt_name=apt_name,
#             docfile=docfile,
#         )
#         fileupload.save()
#         return redirect('fileupload') ##저장한 파일 보내줌
#     else:
#         file_uploadForm= FileUploadForm
#         context={
#             'file_uploadForm': file_uploadForm
#         }
#         return render(request, 'home.html', context)









# def home(request):
#     return render(request, 'home.html')
# def result(request):
#     return render(request, 'result.html')








# def upload_file(request):
#     if request.method == 'POST':
#         form= DocumentForm(request.POST, request.FILES)
#         if form.is_valid():
#             # newdoc= Document(docfile= request.FILES['docfile'])
#             # newdoc.save()
#             handle_uploaded_file(request.FILES['docfile'])
#             return HttpResponseRedirect(reverse('result'))
#         else:
#             form= DocumentForm()
#         # documents= Document.objects.all()
#         return render(request, "home.html", {"form": form})
        