from django.shortcuts import render , redirect
from models import Photo , ImageClass , LabelsClass
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.platform import gfile
import csv
from cart.forms import CartAddProductForm
from math import *
from orders.models import Order
from upload.models import Image


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

def detail(request, image_id):
    det_photo = Photo.objects.get(id=image_id)
    all_photo=Photo.objects.all()
    path1='media/photos'
    path2=det_photo.name
    imagePath1 = os.path.abspath(os.path.join(BASE_DIR, os.path.join(path1,path2)))

    feature = run_on_image(imagePath1)
    searcher = Searcher(path_class)
    results = searcher.search(feature)
    list_images = results
    show = Photo.objects.filter(name__in=results)
    context = {
    'det_photo': det_photo,
    'image_id': image_id,
    'results': results,
    'show' :show,
    'all_photo' :all_photo,}

    return render(request, 'photos/detail.html', context)

def Index(request):
    photos=Photo.objects.order_by('?')[:30]
    cart_product_form = CartAddProductForm()
    return render(request,'photos/index.html',{'photos':photos,'cart_product_form': cart_product_form})

def get_all_classes_and_labels_names():
    all_classes_names = []
    all_classes = ImageClass.objects.all()
    for classX in all_classes:
        all_classes_names.append(classX.class_name)

    all_labels_names = []
    all_labels = LabelsClass.objects.all()
    for label in all_labels:
        all_labels_names.append(label.label_name)

    return all_classes_names , all_labels_names

def get_all_images(name , type):
	if type == 'class':
		related_images = []
		for image_class in ImageClass.objects.all():
			if image_class.class_name == name.lower():
				related_images = image_class.photo_set.all()
				break
	if type == 'label':
		related_images = []
		for image_label in LabelsClass.objects.all():
			if image_label.label_name == name.lower():
				related_images = image_label.photo_set.all()
				break

	return related_images

def search(request):
    all_classes_names , all_labels_names = get_all_classes_and_labels_names()

	# suggested_labels = []
    list_for_each_word = []
    related_images = []
    empty = 1

    class_words_counter = 0
    more_than_one_class = []

    if request.method == "POST":
        search_text = request.POST.get("input")
        search_words = search_text.split()

        for i in range(0 , len(search_words)):
            if search_words[i] in all_labels_names:
                empty = 0

                three_words_label = search_words[i]
                two_words_label = search_words[i]

                if len(search_words) > i+3:
                    three_words_label= search_words[i]+' '+search_words[i+1]+' '+search_words[i+2]
                elif len(search_words) > i+2 :
                    two_words_label = search_words[i]+' '+search_words[i+1]

                if three_words_label in all_labels_names:
                    related_images = get_all_images(three_words_label , 'label')
                    i += 3
                elif two_words_label in all_labels_names:
                    related_images = get_all_images(two_words_label, 'label')
                    i += 2
                else:
                    related_images = get_all_images(search_words[i] , 'label')

                if len(related_images) != 0:
                    list_for_each_word.append(related_images)

            elif search_words[i] in all_classes_names:
                class_words_counter += 1
                more_than_one_class.append(ImageClass.objects.filter(class_name=search_words[i]).first())
                empty = 0
                # suggested_labels = get_labels_for_classX(search_words[i])
                related_images = get_all_images(search_words[i] , 'class')
                if len(related_images) != 0:
					list_for_each_word.append(related_images)

    if len(list_for_each_word) != 0:
            related_images = set(list_for_each_word[0])
            for s in list_for_each_word[1:]:
                related_images.intersection_update(s)


    return render(request, 'photos/search.html' , {'related_images':related_images, 'class_words_counter' : class_words_counter ,
                                                   'more_than_one_class' : more_than_one_class , 'empty':empty,
                                                  'all_classes': ImageClass.objects.all()})

def search_buttons(request , id):
	classX = ImageClass.objects.filter(id=id).first()
	images = get_all_images(classX.class_name , 'class')
	return render(request , 'photos/show_classX_images.html' , {'related_images':images})

def home_page(request , username):
    photos_of_user_orders = Photo.objects.all()
    final_recommended_photos = Photo.objects.all()
    user_uploads =  Image.objects.all()

    if request.user.is_authenticated:
        user_orders = Order.objects.filter(userid = request.user.id)
        list_of_user_orders = []
        for order in user_orders:
            list_of_user_orders.append(order.product.id)
        photos_of_user_orders = Photo.objects.filter(id__in=list_of_user_orders)

        user_uploads = Image.objects.filter(userid = request.user.id)
        list_of_user_uploads = []
        for upload in user_uploads:
            list_of_user_uploads.append(upload.product)
        photos_of_user_uploads = Photo.objects.filter(photo__in=list_of_user_uploads)

        recommended_photos = []
        for photo in photos_of_user_orders:
            feature = run_on_image(photo.photo.url[1:])
            searcher = Searcher(photo.features.url[1:])
            results = searcher.search(feature)
            recommended_photos.append(results)

        flatten_recommended_photos = [item for sublist in recommended_photos for item in sublist]


        final_recommended_photos = Photo.objects.filter(name__in=flatten_recommended_photos)

    return render(request, 'photos/user_home_page.html' , {'photos_of_user_orders':photos_of_user_orders
                                                           , 'orders_size' : len(photos_of_user_orders)
                                                           , 'photos_of_user_uploads' : user_uploads
                                                           , 'uploads_size' : len(user_uploads)
                                                           , 'final_recommended_photos' : final_recommended_photos
                                                           , 'username':request.user.username})




# def imgsearch(request):
#
#     if request.method == 'POST':
#         form = ImageUploadForm(request.POST, request.FILES)
#
#
#         if form.is_valid():
#             # id = uplo
#             # id.product.save(request.FILES['product'].name)
#
#             p = upload1(product=request.FILES['product'])
#             p.save()
#             # return HttpResponseRedirect('/photos/')
#
#     else:
#         form =  ImageUploadForm
#     return render(request, 'upload2/imgsearch.html',{'form':form})
#


    # return render(request, 'photos/imgsearch.html')






    # Do something for anonymous users.



