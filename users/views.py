from django.http import HttpResponseRedirect
from users.forms import RegistrationForm , LoginForm
from django.contrib.auth.models import User
from users.models import UserAccount
from django.contrib.auth import authenticate , login , logout
from django.shortcuts import render,redirect
from photos.views import home_page


def UserRegistration(request):
    if request.user.is_authenticated():
        return HttpResponseRedirect('/photos/')
    elif request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = User.objects.create_user(username=form.cleaned_data['username'],
                                            email=form.cleaned_data['email'],
                                            password=form.cleaned_data['password'])
            user.save()
            user_account = UserAccount(user=user,
                                       username=form.cleaned_data['username'],
                                       email=form.cleaned_data['email'],
                                       password=form.cleaned_data['password'])
            user_account.save()
            return HttpResponseRedirect('/photos/' , {'form' : form})
        else:
            return render(request , 'register.html' , {'form': form})
    else:
        '''user is'nt submitting the form, show them a blank registration form'''
        form = RegistrationForm()
        return render(request,'register.html', {'form' : form})


def LoginRequest(request):
    if request.user.is_authenticated():
        return HttpResponseRedirect('/photos/')
    elif request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user_account = authenticate(username=username , password=password)

            if user_account is not None:
                login(request,user_account)
                return home_page(request , username)

            else:
                return HttpResponseRedirect('/users/login/' , {'form': form})
        else:
            return render(request, 'login.html', {'form': form})
    else:
        '''user is'nt submitting the form, show them a blank registration form'''
        form = LoginForm()
        return render(request,'login.html', {'form' : form})


def LogoutRequest(request):
    logout(request)
    return HttpResponseRedirect('/photos/')