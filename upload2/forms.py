# from django import forms
#
# class ImageUploadForm(forms.Form):
#     """Image upload form."""
#     image = forms.ImageField()
from django import forms
from django.contrib.auth.models import User
from django.forms import ModelForm
from upload2.models import Image1

class ImageUploadForm(forms.ModelForm):
    product = forms.ImageField()

    class Meta:
        model = Image1
        exclude = ('product',)
