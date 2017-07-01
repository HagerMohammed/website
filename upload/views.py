from django.shortcuts import render
from django.contrib.auth.models import User
from .models import Image
from django.http import HttpResponseRedirect
from django.views.decorators.http import require_POST
from django.views import generic
from .forms import ImageForm
from django.contrib.auth.models import User
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import csv
from math import*
import os
import time
import timeit


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# imagePath = os.path.abspath(os.path.join(BASE_DIR, 'img/258890_img_00000002.jpg'))
path_class = os.path.abspath(os.path.join(BASE_DIR, 'media/features'))
modelFullPath = os.path.abspath(os.path.join(BASE_DIR, 'photos/inception_dec_2015/tensorflow_inception_graph.pb'))




def upload_detail(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        idd = request.user.id
        entry = User.objects.get(pk=idd)


        p = Image(userid = entry, product = request.FILES['product'])
        p1=p.product
        context = {
            'p1':p1,
        }
        if form.is_valid():
            p.save()
            return render(request, 'recomendation.html',context)
    else:
        form = ImageForm()
    return render(request, 'details.html',{'form':form})


'''def upload_detail(request):
    return render(request, 'details.html')'''


