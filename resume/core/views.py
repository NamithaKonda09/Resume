from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy
import textract as tx

import glob
import os
import pytesseract
from PIL import Image
import img2pdf

from pytesseract import image_to_string


class Home(TemplateView):
    template_name = 'home.html'


def upload(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
        latest = newest('media')
        text = file_to_string(latest)
        context['text'] = text
    return render(request, 'upload.html', context)


def file_to_string(filepath):
    text = tx.process(filepath)
    text = text.decode('utf-8')

    return text


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)
