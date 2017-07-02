# -*- coding: utf-8 -*-

from .forms import ImageUploadForm
from .models import Image1
from django.http import HttpResponseRedirect
from django.shortcuts import render

from django.views import generic
from django.shortcuts import render , redirect
from django.views.generic.edit import CreateView , UpdateView , DeleteView
from django.http import HttpResponse
import numpy as np
import tensorflow as tf
import argparse
import glob
import os
import re
from tensorflow.python.platform import gfile
import csv
from django.contrib.auth import authenticate , login
from django.views.generic import View
from django.contrib.auth.models import User
from cart.forms import CartAddProductForm
from math import *
from orders.models import Order
from .forms import ImageUploadForm


from django.http import HttpResponseRedirect
from photos import views

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# imagePath = os.path.abspath(os.path.join(BASE_DIR, 'img/258890_img_00000002.jpg'))
path_class = os.path.abspath(os.path.join(BASE_DIR, 'media/features'))
modelFullPath = os.path.abspath(os.path.join(BASE_DIR, 'photos/inception_dec_2015/tensorflow_inception_graph.pb'))


def create_graph():
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_on_image(imagePath):
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    create_graph()

    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        # output = open(indexpath, "w")
        image_data = gfile.FastGFile(imagePath, 'rb').read()
        predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
        queryfeature = np.squeeze(predictions)

    return queryfeature


class Searcher:
    def __init__(self, path_class):

        self.path_class = path_class

    def square_rooted(self, x):
        return round(sqrt(sum([a * a for a in x])), 5)

    def cosine_similarity(self, x, y):
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = self.square_rooted(x) * self.square_rooted(y)
        return round(numerator / float(denominator), 5)
        # tgroba we faslaet bst5dam el numpy

    def cos_sim(self, a, b):

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def euclidean_distance(self, x, y):
        return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

    def manhattan_distance(self, x, y):

        return sum(abs(a - b) for a, b in zip(x, y))

    def jaccard_similarity(self, x, y):

        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality / float(union_cardinality)

    def search(self, queryFeatures, limit=4):

        results = {}
        results1 = {}

        for filename in os.listdir(path_class):
            with open(path_class + '/' + filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    features = [float(x) for x in row[0:]]
                    d = self.cosine_similarity(features, queryFeatures)

                    results[filename[:-4]+'.jpg'] = d

                f.close()
                results1.update(results)


        results1 = sorted(results1, key=results1.get,reverse=True)

        return results1[:limit]

# def handel_on_image(img):
#     features = run_on_image(img.product.url[1:])
#     searcher = Searcher(path_class)
#     results = searcher.search(features)
#     list_images = results
#     # show = Photo.objects.filter(name__in=results)
#     views.detail()
#
#
#     return



def imgsearch(request):

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            p = Image1(product=request.FILES['product'])
            # p = Image1(product=request.FILES('product'))
            p.save()
            # handel_on_image(p)

            # return HttpResponseRedirect('/photo/')
            return render(request, 're.html')



    else:
        form =  ImageUploadForm
    return render(request, 'imgsearch.html',{'form':form})

'''def imgsearch(request):
    return render(request, 'imgsearch.html')'''





