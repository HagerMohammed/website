# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-07-02 03:29
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('photos', '0015_auto_20170701_2330'),
    ]

    operations = [
        migrations.AlterField(
            model_name='upload1',
            name='product',
            field=models.FileField(upload_to='media'),
        ),
    ]