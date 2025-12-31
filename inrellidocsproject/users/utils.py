from django.core.mail import EmailMessage
from rest_framework_simplejwt.tokens import AccessToken
from datetime import timedelta

def generate_email_verification_token(user):
    token = AccessToken.for_user(user)
    token.set_exp(lifetime=timedelta(minutes=10))
    token['type'] = 'email_verification'
    return str(token)


class Util:
    @staticmethod
    def send_mail(data):
        email=EmailMessage(subject=data['email_subject'],body=data['email_body'],to=[data['to_email']])
        email.content_subtype='html'
        email.send()
        