from typing import Generic
from django.db.models import fields
from django.http import request
# from typing_extensions import Required
from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken,TokenError
from .models import CustomUser
from django.contrib.auth import authenticate
from rest_framework.exceptions import AuthenticationFailed
from django.utils.text import gettext_lazy as _
from django.utils.encoding import smart_str,smart_bytes,force_str,DjangoUnicodeDecodeError
from django.utils.http import urlsafe_base64_encode,urlsafe_base64_decode
from django.contrib.sites.shortcuts import get_current_site
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.db import connections
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError




class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(max_length=68, min_length=8, write_only=True)

    # default_error_messages = {
    #     'username': 'The username should only contain alphanumeric characters'}

    class Meta:
        model = CustomUser
        fields = ['email','password']
        
    def validate_password(self, value):#here validate password using custom validators
        try:
            validate_password(value)
        except ValidationError as e:
            raise serializers.ValidationError(e.messages)
        return value

    def create(self, validated_data):
        user = CustomUser(
            email=validated_data['email']
        )
        user.set_password(validated_data['password'])  # ✅ now safe
        user.save()
        return user


    

class LoginSerializer(serializers.ModelSerializer):
    email=serializers.EmailField(max_length=255,min_length=3) #here this validiation email
    password = serializers.CharField(max_length=68, min_length=8, write_only=True)
    
    class Meta:
        model=CustomUser
        fields=['email','password']

    def validate(self,attrs): #here get email and password from user to validate
        email = attrs.get('email')
        password = attrs.get('password')
        print("DBS:", connections.databases)
        print("USING DB:", CustomUser.objects.db)


        try:
            user = CustomUser.objects.get(email=email)
        except CustomUser.DoesNotExist:
            raise AuthenticationFailed("Invalid credentials")

        if not user.check_password(password):
            raise AuthenticationFailed("Invalid credentials")

        if not user.is_active:
            raise AuthenticationFailed("Account disabled")

        attrs['user'] = user
        return attrs


class LogoutSerializer(serializers.Serializer):
    refresh = serializers.CharField()

    def validate(self, attrs):
        self.token = attrs['refresh']
        return attrs

    def save(self):
        try:
            token = RefreshToken(self.token)
            token.blacklist()
        except Exception:
            raise serializers.ValidationError("Invalid or expired token")


class ResendVerificationEmailSerializer(serializers.Serializer):
    email = serializers.EmailField()




class ChangePasswordSerializer(serializers.ModelSerializer):
     old_password=serializers.CharField(max_length=68, min_length=8, write_only=True)
     password=serializers.CharField(max_length=68, min_length=8, write_only=True)
     
     class Meta:
         model=CustomUser
         fields=['old_password','password']



class PasswordEmailResetSerializer(serializers.Serializer):
    email=serializers.EmailField(max_length=2)
    
    class Meta:
        fields=['email']


class PasswordTokenCheckApiSerializer(serializers.Serializer):
    pass



class SetNewPasswordSerializer(serializers.Serializer):
    password = serializers.CharField(
        min_length=8,
        max_length=64,
        write_only=True
    )
    uidb64 = serializers.CharField(write_only=True)
    token = serializers.CharField(write_only=True)

    def validate(self, attrs):
        try:
            password = attrs.get("password")
            uidb64 = attrs.get("uidb64")
            token = attrs.get("token")
            

            user_id = force_str(urlsafe_base64_decode(uidb64))
            user = CustomUser.objects.get(id=user_id)

            if not PasswordResetTokenGenerator().check_token(user, token):
                raise AuthenticationFailed("Reset link is invalid or expired", 401)

            try:
                validate_password(password, user)  # Validate password is used for custom password validators in passwordValidators.py
            except ValidationError as e:
                raise serializers.ValidationError({"password": e.messages})

            user.set_password(password)
            user.save()
            return user

        except Exception as e:
            raise AuthenticationFailed("Reset link is invalid or expired", 401)


class GoogleLoginSerializer(serializers.Serializer):
    token = serializers.CharField()
