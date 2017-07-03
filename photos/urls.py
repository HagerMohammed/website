from django.conf.urls import url
from . import views
from django.conf import settings
from django.views.generic import TemplateView

app_name = 'photos'

urlpatterns = [
	# /photos/
	url(r'^$', views.Index , name='index'),

	# /photos/<photo_id>
	url(r'^(?P<image_id>\d+)/$', views.detail,  name='detail'),

	#/photos/search-results/
	url(r'^search-results/$', views.search, name='search'),

	#/photos/<class_id>
	url(r'^(?P<id>[0-9]+)/$', views.search_buttons , name='search_buttons'),
	# url(r'^search-img/$', views.imgsearch, name='imgsearch'),

	#/photos/user_id
	url(r'^(?P<username>\w+)/$' , views.home_page , name = 'home_page'),

]
