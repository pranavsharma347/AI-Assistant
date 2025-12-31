from django.contrib import admin
from django.urls import path,include
from .views import Register,VerifyEmail,LoginAPI,ChangePassword,PasswordResetEmail,PasswordCheckTokenAPI,SetNewPasswordAPIView,LogoutAPI,ResendVerificationEmail,GoogleLoginAPI


urlpatterns = [
    path('register/',Register.as_view(),name='Register'),
    path('verify-email/',VerifyEmail.as_view(),name='verify-email'),#remember always name contain value is used for reverse url in any function
    path('resend-verification-email/',ResendVerificationEmail.as_view(),name='resend-verification-email'),
    path('login/',LoginAPI.as_view(),name='login'),
    path('changepassword/',ChangePassword.as_view(),name='changepassword'),
    path('password-reset-email/',PasswordResetEmail.as_view(),name='request-reset-email'),
    # path('password-reset/<uidb64>/<token>/',PasswordCheckTokenAPI.as_view(),name='password-reset-confirm'),
    path("password-reset/",PasswordCheckTokenAPI.as_view(),name="password-reset-complete"),
    path('password-reset-complete/',SetNewPasswordAPIView.as_view(),name='password-reset-complete'),
    path('google-login/', GoogleLoginAPI.as_view(), name='google-login'),

    path('logout/',LogoutAPI.as_view(),name='logout'),
]