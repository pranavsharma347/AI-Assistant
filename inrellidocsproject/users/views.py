from django.shortcuts import render
from .models import CustomUser
from .serializers import ChangePasswordSerializer, PasswordEmailResetSerializer, PasswordTokenCheckApiSerializer, RegisterSerializer,LoginSerializer,LogoutSerializer, ResendVerificationEmailSerializer, SetNewPasswordSerializer
from rest_framework import status,generics
from django.contrib.sites.shortcuts import get_current_site
from django.urls import reverse
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken,AccessToken
from .utils import Util, generate_email_verification_token
from rest_framework import views
from django.conf import settings
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.utils.encoding import smart_str,smart_bytes,force_str,DjangoUnicodeDecodeError
from django.utils.http import urlsafe_base64_encode,urlsafe_base64_decode
from .email_templates.email_verification import email_reset_password_body, email_verification_body as email_verification
from datetime import datetime
from rest_framework_simplejwt.authentication import JWTAuthentication
from google.oauth2 import id_token
from google.auth.transport import requests
from rest_framework.response import Response
from django.conf import settings
from .models import CustomUser
from .serializers import GoogleLoginSerializer
from django.db import IntegrityError
import jwt





class Register(generics.GenericAPIView):
    serializer_class = RegisterSerializer

    def post(self, request):
        try:
            email = request.data.get('email')
            password = request.data.get('password')

            if not password:
                return Response(
                    {"success": False, "message": "Password is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            if len(password) < 8:
                return Response(
                    {"success": False, "message": "Password must be at least 8 characters long"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # ✅ Email required check
            if not email:
                return Response(
                    {"success": False, "message": "Email is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # ✅ Check if user already exists
            if CustomUser.objects.filter(email=email).exists():
                return Response(
                    {"success": False, "message": "User with this email already exists"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # ✅ Validate & save user
            serializer = self.serializer_class(data=request.data)
            serializer.is_valid(raise_exception=True)
            serializer.save()

            user = CustomUser.objects.get(email=email)

            # ✅ Generate email verification token
            token = generate_email_verification_token(user)

            # ✅ Frontend verify page URL
            # verify_url = f"http://localhost:5174/verify-email?token={token}"
            verify_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"

            # ✅ Email body (HTML)
            email_body = email_verification(user.email, verify_url)

            # ✅ Send email
            Util.send_mail({
                "email_body": email_body,
                "to_email": user.email,
                "email_subject": "Welcome to IntelliDocs – Verify your email"
            })

            return Response(
                {
                    "success": True,
                    "message": "Registration successful. Please check your email to verify your account."
                },
                status=status.HTTP_201_CREATED
            )

        except IntegrityError:
            return Response(
                {"success": False, "message": "User already exists"},
                status=status.HTTP_400_BAD_REQUEST
            )

        except Exception as e:
            print("Register Error:", str(e))
            return Response(
                {
                    "success": False,
                    "message": "Something went wrong. Please try again later."
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )




class VerifyEmail(APIView):

    def get(self, request):
        token = request.GET.get('token')
        print('token', token)

        if not token:
            return Response(
                {"error": "Token missing"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            payload = AccessToken(token)

            if payload.get('type') != 'email_verification':
                return Response(
                    {"error": "Invalid token type"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            user_id = payload['user_id']
            user = CustomUser.objects.get(id=user_id)

            if user.is_verified:
                return Response(
                    {"message": "Email already verified"},
                    status=status.HTTP_200_OK
                )

            user.is_verified = True
            user.save()

            return Response(
                {"message": "Email verified successfully"},
                status=status.HTTP_200_OK
            )

        except TokenError:
            return Response(
                {"error": "Token expired or invalid"},
                status=status.HTTP_400_BAD_REQUEST
            )

class LoginAPI(generics.GenericAPIView):
    serializer_class=LoginSerializer
    def post(self,request):
        try:
            serializer=self.serializer_class(data=request.data)
            if serializer.is_valid(raise_exception=True):
                token=RefreshToken.for_user(serializer.validated_data['user'])
                user=serializer.validated_data['user']
                if not user.is_verified:
                    user = CustomUser.objects.get(email=user.email)

                    # ✅ Generate email verification token
                    token = generate_email_verification_token(user)

                    # ✅ Frontend verify page URL
                    # verify_url = f"http://localhost:5174/verify-email?token={token}"
                    verify_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"

                    # ✅ Email body (HTML)
                    email_body = email_verification(user.email, verify_url)

                    # ✅ Send email
                    Util.send_mail({
                        "email_body": email_body,
                        "to_email": user.email,
                        "email_subject": "Welcome to IntelliDocs – Verify your email"
                    })

                    return Response(
                        {
                            "success": False,
                            "code": "EMAIL_NOT_VERIFIED",
                            "message": "Please check your email to verify your account."
                        },
                        status=status.HTTP_403_FORBIDDEN
                    )
                print(token)

                return Response({
                "message": "Login successful",
                "user": {
                    "email": user.email,
                },
                "tokens": {
                    "access": str(token.access_token),
                    "refresh": str(token)
                }
        }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error':str(e)},status=status.HTTP_400_BAD_REQUEST)
        

class LogoutAPI(generics.GenericAPIView):
    serializer_class = LogoutSerializer
    permission_classes = [IsAuthenticated]
    # authentication_classes = [JWTAuthentication]  # 👈 MUST

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(
            {"message": "Logout successful"},
            status=status.HTTP_200_OK
        )



class ResendVerificationEmail(generics.GenericAPIView):
    serializer_class = ResendVerificationEmailSerializer   # You can use any serializer here since we only need the email field

    def post(self, request):
        email = request.data.get('email')
        if not email:
            return Response({'error': 'Email is required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            user = CustomUser.objects.get(email=email)
            if user.is_verified:
                return Response({'message': 'Email is already verified'}, status=status.HTTP_200_OK)

            token = generate_email_verification_token(user)
            # ✅ Frontend verify page URL
            # verify_url = f"http://localhost:5174/verify-email?token={token}"
            verify_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"

            # ✅ Email body (HTML)
            email_body = email_verification(user.email, verify_url)

            # ✅ Send email
            Util.send_mail({
                "email_body": email_body,
                "to_email": user.email,
                "email_subject": "Welcome to IntelliDocs – Verify your email"
            })

            return Response({"message": "Verification email resent. Please check your email."}, status=status.HTTP_200_OK)

        except CustomUser.DoesNotExist:
            return Response({'error': 'User with this email does not exist'}, status=status.HTTP_404_NOT_FOUND)

        


class ChangePassword(generics.UpdateAPIView):
    permission_classes=(IsAuthenticated,)
    serializer_class=ChangePasswordSerializer
    
        
    def update(self,request):
        
            data=CustomUser.objects.get(email=request.user.email)
            serializer=self.serializer_class(data=request.data)
            if serializer.is_valid():
                print('working')
                if not data.check_password(serializer.data.get('old_password')):
                     return Response({'message':"invalid old password"})
                    
                data.set_password(serializer.data.get('password'))
                data.save()
                return Response({'message':"new password set successfully"})
            return Response({"hai":"hai"})
    


class PasswordResetEmail(generics.GenericAPIView):
    serializer_class=PasswordEmailResetSerializer
    
    def post(self,request):
        serializer=self.serializer_class(data=request.data)
        email=request.data['email']
        if CustomUser.objects.filter(email=email).exists():#now from here we have to token based email to send user
            user=CustomUser.objects.get(email=email)
            uidb64=urlsafe_base64_encode(smart_bytes(user.id))
            #now we create a token for the user
            token=PasswordResetTokenGenerator().make_token(user)
            # verify_url = f"http://localhost:5174/reset-password/{uidb64}/{token}"
            # verify_url = f"{settings.FRONTEND_URL}/reset-password/{uidb64}/{token}"
            verify_url = f"{settings.FRONTEND_URL}/reset-password?uidb64={uidb64}&token={token}"

            # ✅ Email body (HTML)
            email_body = email_reset_password_body(user.email, verify_url)

            # ✅ Send email
            Util.send_mail({
                "email_body": email_body,
                "to_email": user.email,
                "email_subject": "Welcome to IntelliDocs – Reset your password"
            })

            return Response({"message": "Verification email resent. Please check your email."}, status=status.HTTP_200_OK)
        return Response({'error':'User with this email does not exists'},status=status.HTTP_404_NOT_FOUND)
        


      
class PasswordCheckTokenAPI(generics.GenericAPIView):
    Serializer_class=PasswordTokenCheckApiSerializer

    def get(self,request):#remember here we pass uid64,token then we have to pass in url
        try:
            uidb64 = request.GET.get('uidb64')
            token = request.GET.get('token')
            id=smart_str(urlsafe_base64_decode(uidb64))
            print('id',id)
            user=CustomUser.objects.get(id=id)
            
            if not PasswordResetTokenGenerator().check_token(user,token):
                return Response({'Error':'Token is invalid,please request is new one'},status=status.HTTP_401_UNAUTHORIZED)
            return Response({"success":True,",message":"Credentials is valid","uidb64":uidb64,"token":token},status=status.HTTP_200_OK)
            
            
            
        except Exception as e:
            return Response(e)
        


class SetNewPasswordAPIView(generics.GenericAPIView):
    serializer_class=SetNewPasswordSerializer
    
    def put(self,request):
        serializer=self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        return Response({'success':True,"message":"Password Reset Successfully"},status=status.HTTP_200_OK)
    


class GoogleLoginAPI(generics.GenericAPIView):
    serializer_class = GoogleLoginSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)

        token = serializer.validated_data['token']
        print("token",token)

        try:
            # 🔐 Verify Google token
            idinfo = id_token.verify_oauth2_token(
                token,
                requests.Request(),
                settings.GOOGLE_CLIENT_ID
            )

            email = idinfo['email']
            name = idinfo.get('name', '')

            user, created = CustomUser.objects.get_or_create(
                email=email,
                defaults={
                    "is_verified": True,
                }
            )

            user.is_verified = True
            user.save()

            refresh = RefreshToken.for_user(user)

            return Response({
                "success": True,
                "message": "Google login successful",
                "user": {
                    "email": user.email
                },
                "tokens": {
                    "access": str(refresh.access_token),
                    "refresh": str(refresh)
                }
            }, status=status.HTTP_200_OK)

        except ValueError:
            return Response(
                {"error": "Invalid Google token"},
                status=status.HTTP_400_BAD_REQUEST
            )