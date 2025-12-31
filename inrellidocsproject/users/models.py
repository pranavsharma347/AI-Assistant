from django.db import models

from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.db import models
from .managers import CustomUserManager
from django.utils.translation import gettext_lazy as _
from rest_framework_simplejwt.tokens import RefreshToken

class CustomUser(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(_('email address'), unique=True)
    is_verified = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
    objects = CustomUserManager()  # ✅ REQUIRED



    def __str__(self):
        return self.email
    

    
    def tokens(self):
        refresh=RefreshToken.for_user(self) # here when user login it creates a token for user and this same token will be retun by login seralizer
        return{
            'refresh':str(refresh),
            'access':str(refresh.access_token)
        }