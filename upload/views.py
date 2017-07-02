from django.shortcuts import render
from .models import Image
from django.contrib.auth.models import User
import os
from photos.models import Photo , ImageClass
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from django.core.files import File
import csv
from itertools import izip
from django.http import HttpResponseRedirect
from.forms import ImageForm



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
modelFullPath_inception = os.path.abspath(os.path.join(BASE_DIR, 'photos/inception_dec_2015/tensorflow_inception_graph.pb'))
# # imagePath = os.path.abspath(os.path.join(BASE_DIR, 'img/258890_img_00000002.jpg'))
# path_class = os.path.abspath(os.path.join(BASE_DIR, 'media/features'))
modelFullPath = os.path.abspath(os.path.join(BASE_DIR, 'models/deepfasion_v3.pb'))
modelFullPath_multilabeling1 =  os.path.abspath(os.path.join(BASE_DIR , 'model.h5'))
modelFullPath_multilabeling2 =  os.path.abspath(os.path.join(BASE_DIR , 'model.json'))




def create_graph():
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_on_image1(img_path):
    # pic=img()
    # pic.photo=img.product
    # image=img.product

    answer = None

    if not tf.gfile.Exists(img_path):
        tf.logging.fatal('File does not exist %s', img_path)
        return answer

    image_data = tf.gfile.FastGFile(img_path, 'rb').read()


    create_graph()

    with tf.Session() as sess:

        next_to_last_tensor=sess.graph.get_tensor_by_name('final_result:0')
        image_data = gfile.FastGFile(img_path, 'rb').read()
        predictions = sess.run(next_to_last_tensor,{'DecodeJpeg/contents:0': image_data})
        queryfeature = np.squeeze(predictions)

    return queryfeature


def run_on_image_features(imagePath):
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

def handling_uploaded_photo(img):
    pic_name = img.product.url[13:-4]

    pic = Photo()
    pic.name = pic_name + '.jpg'
    pic.class_name = ImageClass.objects.get(class_name = 'temp')

    pic.photo.save(pic_name + '.jpg', File(open(img.product.url[1:] , 'r')))

    # feature extraction
    features = run_on_image_features(img.product.url[1:])
    f = np.asarray(features)
    np.savetxt('upload/temp_features/' + pic_name + '.csv', f , delimiter=",")
    a = izip(*csv.reader(open('upload/temp_features/' + pic_name + '.csv', "rb")))
    csv.writer(open('upload/temp_features/' + pic_name + '.csv', "wb")).writerows(a)
    pic.features.save(pic_name+'.csv' , File(open('upload/temp_features/' + pic_name + '.csv' , 'r')))

    res = run_on_image1(img.product.url[1:])
    max = 0
    ii = 0
    for i, acc in enumerate(res):
        if acc > max:
            max = acc
            ii = i
    print ii
    f = open('models/deepfasion_labels_v3.txt', "r")
    lines = f.readlines()
    pic_class = lines[ii]

    print pic_class[:-1]


    pic.class_name = ImageClass.objects.get(class_name=pic_class[:-1])
    pic.save()
    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    #
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("model.h5")
    # print("Loaded model from disk")
    #
    # # evaluate loaded model on test data
    # loaded_model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    #
    # prediction = loaded_model.predict(np.reshape(f, (1, -1)))
    # pred1 = prediction > np.max(prediction) - 0.2 * np.max(prediction)
    #
    # list_of_labels = get_labels(pred1)
    # print list_of_labels
    #
    # for label in list_of_labels:
    #     l = LabelsClass.objects.get(label_name=label)
    #     pic.label_name.add(l)
    # pic.save()
    #
    # pic.class_name = 'pantalon'
    # pic.save()


def get_labels(predictions):
    f = open('list_attr_cloth.txt')
    lines = f.readlines()

    arr = np.array(predictions)
    line_in_a_list = (arr.tolist())[0]

    list_of_index = []
    offset = 0
    while 1:
        if True in line_in_a_list:
            index = line_in_a_list.index(True)

            words_in_a_line = (lines[index + offset]).split()
            label = ""
            for i in range(0, len(words_in_a_line) - 1):
                if i == len(words_in_a_line) - 2:
                    label += (words_in_a_line[i])
                else:
                    label += (words_in_a_line[i] + " ")
            list_of_index.append(label)

            offset += 1
            line_in_a_list.pop(index)
        else:
            break

    return  list_of_index




def upload_detail(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        idd = request.user.id
        entry = User.objects.get(pk=idd)
        p = Image(userid = entry, product = request.FILES['product'])

        if form.is_valid():
            p.save()
            handling_uploaded_photo(p)
            return HttpResponseRedirect('/photos/')

    else:
        form = ImageForm()
    return render(request, 'details.html',{'form':form})


# def upload_pic(request):
#     if request.method == 'POST':
#         form = ImageUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             m = ExampleModel.objects.get(pk=course_id)
#             m.model_pic = form.cleaned_data['image']
#             m.save()
#             return HttpResponse('image upload success')
#     return HttpResponseForbidden('allowed only via POST')



'''def upload_detail(request):
    return render(request, 'details.html')'''


