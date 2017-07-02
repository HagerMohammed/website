# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

class Image1(models.Model):
    product = models.ImageField(upload_to='media')